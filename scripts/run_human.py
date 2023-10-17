import sys

from CarEnv.Configs import get_standard_env_config, get_standard_env_names
from CarEnv.Env import CarEnv
from time import time, sleep


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--env', choices=get_standard_env_names(), default='racing')
    args = parser.parse_args()

    cfg = get_standard_env_config(args.env)
    cfg['action'] = {'type': 'human'} if args.env == 'parking' else {'type': 'human_pedals'}
    cfg['dt'] = 0.05
    env = CarEnv(cfg, render_mode='human', render_kwargs={'fullscreen': True, 'hints': {'scale': 25.}})
    env.reset(seed=0)

    while True:
        done = False
        total_reward = 0

        while not done:
            curr_time = time()
            act = env.action_space.sample()
            obs, rew, terminated, truncated, *_ = env.step(act)
            done = terminated or truncated
            total_reward += rew

            # Dynamically set the time delta based on actual refresh rate.
            # This is generally not recommended during serious use but allows
            # for nicer visuals for this demo.
            env.dt = time() - curr_time

        print(f"{total_reward = }", file=sys.stderr)
        env.reset()


if __name__ == '__main__':
    main()
