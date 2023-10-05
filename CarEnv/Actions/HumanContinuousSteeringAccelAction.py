from typing import Tuple, Any

import gymnasium as gym
import numpy as np

from .Action import Action
from ..Physics.VelocityController import SimpleEngineDragVelocityController


class HumanContinuousSteeringAccelAction(Action):
    def __init__(self, js_steer_axis=0, js_throttle_axis=2, js_brake_axis=3):
        import pygame

        self.lat_control = None
        self.long_control = None
        self.js_steer_axis = js_steer_axis
        self.js_throttle_axis = js_throttle_axis
        self.js_brake_axis = js_brake_axis

        pygame.init()

        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
        else:
            self.joystick = None

    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Box(-1, 1, shape=(2,))

    def configure(self, latitudinal_controller, longitudinal_controller):
        if not isinstance(longitudinal_controller, (SimpleEngineDragVelocityController,)):
            raise TypeError(f"{type(longitudinal_controller)}")

        self.lat_control = latitudinal_controller
        self.long_control = longitudinal_controller

    def interpret(self, act) -> Tuple[Any, Any]:
        import pygame
        from pygame.locals import K_LEFT, K_RIGHT, K_UP, K_DOWN

        # Ignore given action and read keyboard

        if self.joystick is not None:
            s = np.clip(self.joystick.get_axis(self.js_steer_axis) / .25, -1, 1)

            throttle_position = 1 - (self.joystick.get_axis(self.js_throttle_axis) + 1) / 2
            brake_position = 1 - (self.joystick.get_axis(self.js_brake_axis) + 1) / 2

            t = throttle_position - brake_position
        else:
            pressed = pygame.key.get_pressed()

            s = 0.0
            t = 0.0

            if pressed[K_LEFT]:
                s -= 1.0
            if pressed[K_RIGHT]:
                s += 1.0
            if pressed[K_UP]:
                t += 1.0
            if pressed[K_DOWN]:
                t -= 1.0

        return s * self.lat_control.max_angle, np.array([t])
