from typing import Tuple, Any

import gymnasium as gym
import numpy as np

from .Action import Action
from ..Physics.VelocityController import SimpleEngineDragVelocityController


class ContinuousSteeringAccelAction(Action):
    def __init__(self):
        self.lat_control = None
        self.long_control = None

    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Box(-1, 1, shape=(2,))

    def configure(self, latitudinal_controller, longitudinal_controller):
        self.lat_control = latitudinal_controller
        self.long_control = longitudinal_controller
        assert isinstance(self.long_control, (SimpleEngineDragVelocityController,))

    def interpret(self, act) -> Tuple[Any, Any]:
        s_act, t_act = act
        s = np.clip(s_act, -1, 1) * self.lat_control.max_angle
        t = np.clip(t_act, -1, 1)
        return s, np.array([t])
