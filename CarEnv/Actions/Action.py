from typing import Tuple, Any
import gym


class Action:
    @property
    def action_space(self) -> gym.Space:
        raise NotImplementedError

    def configure(self, latitudinal_controller, longitudinal_controller):
        pass

    def interpret(self, act) -> Tuple[Any, Any]:
        # Interpret action, return output for latitudinal and longitudinal controllers
        raise NotImplementedError
