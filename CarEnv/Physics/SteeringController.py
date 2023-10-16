import numpy as np
from typing import Tuple


class AbstractSteeringModel:
    @property
    def beta(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def integrate(self, control, dt):
        raise NotImplementedError


class DirectSteeringModel(AbstractSteeringModel):
    def __init__(self, beta_max: float):
        self.beta_max = beta_max
        self._beta = 0.

    @property
    def beta(self):
        return self._beta

    def reset(self):
        self._beta = 0.

    def integrate(self, control, dt):
        self._beta = np.clip(control * self.beta_max, -self.beta_max, self.beta_max)


class RateLimitedSteeringModel(AbstractSteeringModel):
    def __init__(self, beta_max: float, beta_rate: float):
        self.beta_max = beta_max
        self.beta_rate = beta_rate
        self._beta = 0.

    @property
    def beta(self):
        return self._beta

    def reset(self):
        self._beta = 0.

    def integrate(self, control, dt):
        assert np.shape(control) == ()
        beta_target = np.clip(control * self.beta_max, -self.beta_max, self.beta_max)
        d_beta = np.sign(beta_target - self._beta) * self.beta_rate
        new_beta = np.clip(self._beta + dt * d_beta, -self.beta_max, self.beta_max)
        zero_crossing = np.sign(beta_target - self._beta) != np.sign(beta_target - new_beta)
        if zero_crossing:
            new_beta = beta_target
        self._beta = new_beta
