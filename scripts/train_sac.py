import os
import sys
from typing import Optional, Dict

from stable_baselines3.sac import SAC
from stable_baselines3.dqn.policies import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
import torch
import numpy as np


def aggregate_sum(x, indices, n_bins):
    x_agg = torch.zeros((n_bins, x.shape[-1]), device=x.device)
    if indices.numel() > 0:
        torch.index_put_(x_agg, (indices,), x, accumulate=True)
    return x_agg


class ItemEncoder(torch.nn.Module):
    def __init__(self, mlp):
        super(ItemEncoder, self).__init__()
        self.mlp = mlp

    def forward(self, x, idxs, n_bins):
        items_encoded = self.mlp(x)
        embeddings = aggregate_sum(items_encoded, idxs, n_bins)
        return embeddings


def make_item_encoder_network(in_feature, net_arch):
    whole_arch = [in_feature] + net_arch
    layers = []

    for in_dim, out_dim in zip(whole_arch[:-1], whole_arch[1:]):
        layers.append(torch.nn.Linear(in_dim, out_dim))
        layers.append(torch.nn.ReLU())

    return torch.nn.Sequential(*layers)


class PassthroughModel(torch.nn.Module):
    def forward(self, x):
        return x


def make_static_encoder(space, static_dims):
    from torch.nn import Sequential, ReLU, Linear

    if len(space.shape) != 1:
        raise ValueError(f"Unsupported static space: {space}")

    if static_dims == 0:
        # Dummy
        return PassthroughModel(), space.shape[0]
    else:
        return Sequential(Linear(space.shape[0], static_dims), ReLU()), static_dims


class SetFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, static_dims=0, item_arch: Optional[Dict[str, int]] = None,
                 set_dims: Optional[Dict[str, int]] = None) -> None:
        super(SetFeatureExtractor, self).__init__(observation_space, 1)

        if item_arch is None:
            item_arch = dict()
        else:
            # Shallow copy because modified later
            item_arch = {k: v for k, v in item_arch.items()}

        if set_dims is None:
            set_dims = dict()

        total_features = 0

        item_encoders = {}
        set_encoders = {}

        for key, space in observation_space.spaces.items():
            if "_set" in key:
                max_obs, obs_features = space.shape

                arch = item_arch.pop(key, [64, 128])
                set_dim = set_dims.get(key, arch[-1])

                item_encoder_mlp = make_item_encoder_network(obs_features - 1, arch)

                set_encoder = torch.nn.Sequential(
                    torch.nn.Linear(arch[-1], set_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(set_dim, set_dim),
                    torch.nn.ReLU(),
                )

                item_encoders[key] = ItemEncoder(item_encoder_mlp)
                set_encoders[key] = set_encoder
                total_features += set_dim
            else:
                item_encoders[key], dims = make_static_encoder(space, static_dims)
                total_features += dims

        # Fail if not empty because it usually is an error (e.g. typo)
        if item_arch:
            raise ValueError(f"Unused item architectures: {list(item_arch.keys())}")

        self._features_dim = total_features
        self.item_encoders = torch.nn.ModuleDict(item_encoders)
        self.set_encoders = torch.nn.ModuleDict(set_encoders)

    @staticmethod
    def batchify_set(values):
        batch_size, max_obs, obs_features = values.shape

        # Discard all items which do not have the "present" flag in the obs_feature
        # The present flag must always be the first feature!
        all_batch_idxs = torch.broadcast_to(torch.arange(batch_size, device=values.device)[..., None],
                                            (batch_size, max_obs))
        keep_mask = values[..., 0] > 0
        # Mask and discard the present flag
        items = values[keep_mask][..., 1:].reshape(-1, obs_features - 1)
        batch_idxs = all_batch_idxs[keep_mask].flatten()

        return items, batch_idxs, batch_size

    def forward(self, obs):
        result = []

        for key, encoder in self.item_encoders.items():
            if "_set" in key:
                items, batch_idxs, batch_size = self.batchify_set(obs[key])
                items_enc_agg = encoder(items, batch_idxs, batch_size)
                set_enc = self.set_encoders[key](items_enc_agg)
                result.append(set_enc)
            else:
                result.append(encoder(obs[key]))

        return torch.cat(result, dim=-1)


class EvalCallback(BaseCallback):
    def __init__(self, env, freq=50_000, tries=10):
        super(EvalCallback, self).__init__()
        self.freq = freq
        self.tries = tries
        self.env = env

    def _on_step(self) -> bool:
        import wandb

        if self.num_timesteps % self.freq != 0:
            return True

        episode_rewards = []
        episode_steps = []
        bitmaps = []

        for i in range(self.tries):
            obs, _ = self.env.reset(seed=i)
            if i == 0:
                bitmaps.append(self.env.render())
            done = False
            acc_rew = 0.
            step = 0
            while not done:
                act, _ = self.model.predict(obs, deterministic=True)
                obs, rew, terminated, truncated, info = self.env.step(act)
                done = terminated or truncated
                if i == 0:
                    bitmaps.append(self.env.render())
                acc_rew += rew
                step += 1
            episode_rewards.append(acc_rew)
            episode_steps.append(step)

        bitmaps = np.transpose(np.stack(bitmaps), (0, 3, 1, 2))
        dt = self.env.unwrapped.dt if isinstance(self.env, gym.Wrapper) else self.env.dt
        fps = int(np.ceil(1 / dt))
        wandb.log({"eval/video": wandb.Video(bitmaps, fps=fps)}, step=self.num_timesteps)
        self.logger.record("eval/ep_rew_mean", np.mean(episode_rewards))
        self.logger.record("eval/ep_len_mean", np.mean(episode_steps))

        return True


def main():
    from argparse import ArgumentParser
    import wandb
    from stable_baselines3.common.callbacks import ProgressBarCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
    from CarEnv import get_registered

    parser = ArgumentParser()
    parser.add_argument('--gamma', type=float, default=.97)
    parser.add_argument('--static_dims', default=0, type=int)
    parser.add_argument('--timesteps', type=int, default=1_000_000)
    parser.add_argument('--buffer_size', type=int, default=200_000)
    parser.add_argument('--env', choices=get_registered(), default=get_registered()[0])
    args = parser.parse_args()

    env = gym.make(args.env)
    eval_env = gym.make(args.env, render_mode='rgb_array')

    os.makedirs('./tmp', exist_ok=True)
    wandb.init(project="CarEnv", resume='never', config=args, sync_tensorboard=True, dir='./tmp')

    policy_kwargs = {
        'features_extractor_class': SetFeatureExtractor,
        'net_arch': [256, 256],
        'features_extractor_kwargs': {
            'item_arch': {
                "cones_set": [64, 32, 128],
            },
            'static_dims': args.static_dims,
        }
    }

    print(f"{args = }", file=sys.stderr)
    print(f"{policy_kwargs = }", file=sys.stderr)
    print(f"{torch.cuda.is_available() = }", file=sys.stderr)

    act_shape = env.action_space.shape
    noise = OrnsteinUhlenbeckActionNoise(np.zeros(act_shape), np.ones(act_shape) * .2, dt=1e-1) if args.env == 'racing' else None
    agent = SAC("MultiInputPolicy", env, buffer_size=args.buffer_size, action_noise=noise, gamma=args.gamma,
                policy_kwargs=policy_kwargs)

    agent.set_logger(configure(wandb.run.dir, ["tensorboard"]))
    agent.learn(args.timesteps, callback=[ProgressBarCallback(), EvalCallback(eval_env)])


if __name__ == '__main__':
    main()
