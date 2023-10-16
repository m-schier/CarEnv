import gymnasium as gym

from .Action import Action
from ..Physics.VelocityController import DirectVelocityController, LinearVelocityController
from typing import Any, Tuple
import numpy as np


class ContinuousSteeringPedalsAction(Action):
    def __init__(self):
        self.throttle_position_ = 0.
        self.brake_position_ = 0.

    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Box(-1, 1, shape=(3,))

    def interpret(self, act) -> Tuple[Any, Any]:
        s_act, t_act, b_act = act
        s = np.clip(s_act, -1, 1)
        # Map pedals: [-.9, .9] -> [0, 1] to introduce dead zones
        self.throttle_position_ = np.clip(t_act / 1.8 + .5, 0, 1)
        self.brake_position_ = np.clip(b_act / 1.8 + .5, 0, 1)
        return s, np.array([self.throttle_position_, self.brake_position_])
