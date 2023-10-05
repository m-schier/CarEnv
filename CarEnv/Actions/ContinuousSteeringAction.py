import gymnasium as gym

from .Action import Action
from ..Physics.VelocityController import DirectVelocityController, LinearVelocityController
from typing import Any, Tuple
import numpy as np


class ContinuousSteeringAction(Action):
    def __init__(self, target_speed=15 / 3.6):
        self.target_speed = target_speed
        self.lat_control = None
        self.long_control = None

    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Box(-1, 1, shape=(1,))

    def configure(self, latitudinal_controller, longitudinal_controller):
        self.lat_control = latitudinal_controller
        self.long_control = longitudinal_controller
        assert isinstance(self.long_control, (DirectVelocityController, LinearVelocityController))

    def interpret(self, act) -> Tuple[Any, Any]:
        act, = act
        s = np.clip(act, -1, 1) * self.lat_control.max_angle
        return s, self.target_speed
