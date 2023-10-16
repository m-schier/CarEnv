import sys
from typing import Tuple, Any

import gymnasium as gym
import numpy as np

from .Action import Action


class HumanContinuousSteeringPedalsAction(Action):
    def __init__(self, js_steer_axis=0, js_throttle_axis=2, js_brake_axis=3):
        import pygame

        self.js_steer_axis = js_steer_axis
        self.js_throttle_axis = js_throttle_axis
        self.js_brake_axis = js_brake_axis

        pygame.init()

        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            print(f"HumanContinuousSteeringPedalsAction: Using joystick 0 with {self.joystick.get_numaxes()} axes and {self.joystick.get_numbuttons()} buttons", file=sys.stderr)
        else:
            self.joystick = None
            print("HumanContinuousSteeringPedalsAction: Using keyboard", file=sys.stderr)

        self.throttle_position_ = 0.
        self.brake_position_ = 0.

    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Box(-1, 1, shape=(2,))

    def interpret(self, act) -> Tuple[Any, Any]:
        import pygame
        from pygame.locals import K_LEFT, K_RIGHT, K_UP, K_DOWN

        # Ignore given action and read keyboard

        if self.joystick is not None:
            s = np.clip(self.joystick.get_axis(self.js_steer_axis) / .25, -1, 1)

            self.throttle_position_ = 1 - (self.joystick.get_axis(self.js_throttle_axis) + 1) / 2
            self.brake_position_ = 1 - (self.joystick.get_axis(self.js_brake_axis) + 1) / 2
        else:
            pressed = pygame.key.get_pressed()

            s = 0.0

            if pressed[K_LEFT]:
                s -= 1.0
            if pressed[K_RIGHT]:
                s += 1.0

            self.throttle_position_ = 1. if pressed[K_UP] else 0.
            self.brake_position_ = 1. if pressed[K_DOWN] else 0.

        return s, np.array([self.throttle_position_, self.brake_position_])
