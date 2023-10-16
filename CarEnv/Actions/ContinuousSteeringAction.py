import gymnasium as gym

from .Action import Action
from typing import Any, Tuple
import numpy as np


class ContinuousSteeringAction(Action):
    def __init__(self, target_speed=15 / 3.6):
        self.target_speed = target_speed

    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Box(-1, 1, shape=(1,))

    def interpret(self, act) -> Tuple[Any, Any]:
        act, = act
        s = np.clip(act, -1, 1)
        return s, self.target_speed
