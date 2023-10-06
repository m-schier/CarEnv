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
    cfg['action'] = {'type': 'human_pedals'}
    cfg['dt'] = 0.05
    env = CarEnv(cfg)

    while True:
        env.reset()
        done = False
        total_reward = 0

        while not done:
            curr_time = time()
            act = env.action_space.sample()
            obs, rew, terminated, truncated, *_ = env.step(act)
            done = terminated or truncated
            total_reward += rew
            env.render()

            delta = time() - curr_time + env.dt

            if delta > 0:
                sleep(delta)

        print(f"{total_reward = }", file=sys.stderr)


if __name__ == '__main__':
    main()
