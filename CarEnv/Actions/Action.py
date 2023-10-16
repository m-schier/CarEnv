from typing import Tuple, Any
import gymnasium as gym


class Action:
    @property
    def action_space(self) -> gym.Space:
        raise NotImplementedError

    def interpret(self, act) -> Tuple[Any, Any]:
        # Interpret action, return output for latitudinal and longitudinal controllers
        raise NotImplementedError

    def render(self, ctx, width, height):
        pass
