from typing import Tuple, Any

import gymnasium as gym
import numpy as np

from .Action import Action


class ContinuousSteeringAccelAction(Action):
    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Box(-1, 1, shape=(2,))

    def interpret(self, act) -> Tuple[Any, Any]:
        s_act, t_act = act
        s = np.clip(s_act, -1, 1)
        t = np.clip(t_act, -1, 1)
        return s, np.array([t])
