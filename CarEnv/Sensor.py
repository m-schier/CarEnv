import gym


class Sensor:
    def __init__(self, env):
        self.env = env

    @property
    def observation_space(self) -> gym.Space:
        raise NotImplementedError

    def observe(self, env):
        raise NotImplementedError
