import numpy as np
from typing import Tuple


class AbstractSteeringController:
    def __init__(self, max_angle: float):
        """
        Abstract steering controller
        :param max_angle: maximum steering angle in both directions in radians, must be nonnegative
        """
        self.max_angle = max_angle

    @property
    def state_size(self) -> int:
        raise NotImplementedError

    def get_angle(self, state):
        raise NotImplementedError

    def initialize_state(self, n: int):
        return np.zeros((n, self.state_size))

    def update(self, state: np.ndarray, control: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError


class DirectSteeringController(AbstractSteeringController):
    def __init__(self, max_angle: float):
        super(DirectSteeringController, self).__init__(max_angle)

    @property
    def state_size(self) -> int:
        return 1

    def get_angle(self, state):
        return state.flatten()

    def update(self, state: np.ndarray, control: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Faster than clip
        s_new = np.maximum(-self.max_angle, np.minimum(self.max_angle, control))
        return s_new[:, None], s_new, s_new


class LinearSteeringController(AbstractSteeringController):
    def __init__(self, max_angle: float, rate: float):
        super(LinearSteeringController, self).__init__(max_angle)
        self.rate = rate

    @property
    def state_size(self) -> int:
        return 1

    def update(self, state: np.ndarray, control: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Limit control to allowed range
        state = state.flatten()
        control = np.clip(control, -self.max_angle, self.max_angle)

        # Calculate the new angle as limited by the rate of change
        delta_new = np.clip(control, state - self.rate * dt, state + self.rate * dt)

        # Calculate the time taken to reach the new angle as fraction of delta time
        ttr = np.abs(delta_new - state) / self.rate / dt

        # Calculate the mean angle during the time step based on the time taken to reach the new angle
        delta_avg = (delta_new + state) / 2 * ttr + delta_new * (1 - ttr)

        # Also limit to no speed and max speed
        return delta_new[:, None], delta_new, delta_avg

    def get_angle(self, state: np.ndarray):
        return state.flatten()
