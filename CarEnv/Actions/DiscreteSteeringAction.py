import gymnasium as gym

from .Action import Action
from ..Physics.VelocityController import DirectVelocityController, LinearVelocityController
from typing import Any, Tuple
import numpy as np


class DiscreteSteeringAction(Action):
    def __init__(self, target_speed=15 / 3.6, n_actions=3):
        self.target_speed = target_speed
        self.lat_control = None
        self.long_control = None
        self.lut = np.linspace(-1, 1, n_actions)

    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Discrete(len(self.lut))

    def configure(self, latitudinal_controller, longitudinal_controller):
        self.lat_control = latitudinal_controller
        self.long_control = longitudinal_controller
        assert isinstance(self.long_control, (DirectVelocityController, LinearVelocityController))

    def interpret(self, act) -> Tuple[Any, Any]:
        s = self.lut[act] * self.lat_control.max_angle
        return s, self.target_speed
