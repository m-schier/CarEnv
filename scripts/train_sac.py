import os
from typing import Optional, Dict

from stable_baselines3.sac import SAC
from stable_baselines3.dqn.policies import BaseFeaturesExtractor
import gymnasium as gym
import torch


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
    if len(space.shape) != 1:
        raise ValueError(f"Unsupported static space: {space}")

    if static_dims == 0:
        # Dummy
        return PassthroughModel(), space.shape[0]
    else:
        return torch.nn.Sequential(torch.nn.Linear(space.shape[0], static_dims), torch.nn.ReLU()), static_dims


class SetFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, static_dims=0, item_arch: Optional[Dict[str, int]] = None,
                 set_dims: Optional[Dict[str, int]] = None) -> None:
        super(SetFeatureExtractor, self).__init__(observation_space, 1)

        if item_arch is None:
            item_arch = dict()

        if set_dims is None:
            set_dims = dict()

        total_features = 0

        item_encoders = {}
        set_encoders = {}

        for key, space in observation_space.spaces.items():
            if "_set" in key:
                max_obs, obs_features = space.shape

                arch = item_arch.get(key, [64, 128])
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


def main():
    from argparse import ArgumentParser
    import wandb
    from stable_baselines3.common.callbacks import ProgressBarCallback
    from stable_baselines3.common.logger import configure
    from CarEnv.Configs import get_standard_env_config, get_standard_env_names
    from CarEnv.Env import CarEnv

    parser = ArgumentParser()
    parser.add_argument('--gamma', type=float, default=.97)
    parser.add_argument('--timesteps', type=int, default=1_000_000)
    parser.add_argument('--env', choices=get_standard_env_names(), default=get_standard_env_names()[0])
    args = parser.parse_args()

    env = CarEnv(get_standard_env_config(args.env))

    os.makedirs('./tmp', exist_ok=True)
    wandb.init(project="CarEnv", resume='never', config=args, sync_tensorboard=True, dir='./tmp')

    policy_kwargs = {
        'features_extractor_class': SetFeatureExtractor,
        'net_arch': [256, 256, 256],
        'features_extractor_kwargs': {
            'item_arch': {
                "track_set": [64, 32, 128],
            }
        }
    }

    agent = SAC("MultiInputPolicy", env, buffer_size=200_000, gamma=args.gamma, policy_kwargs=policy_kwargs)

    agent.set_logger(configure(wandb.run.dir, ["tensorboard"]))
    agent.learn(args.timesteps, callback=[ProgressBarCallback()])


if __name__ == '__main__':
    main()
