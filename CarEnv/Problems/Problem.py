from typing import Tuple


class Problem:
    @property
    def state_observation_space(self):
        raise NotImplementedError

    def observe_state(self, env):
        raise NotImplementedError

    def configure_env(self, env, rng=None) -> Tuple[float, float, float]:
        raise NotImplementedError

    def render(self, ctx, env):
        pass

    def update(self, env, dt: float) -> Tuple[bool, bool]:
        """
        Update the environment for the problem
        Args:
            env: Environment
            dt: Time delta in seconds

        Returns:
            terminate, truncate
        """
        raise NotImplementedError
