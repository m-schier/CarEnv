import gym
import numpy as np
from .FreeDriveProblem import FreeDriveProblem


class RacingProblem(FreeDriveProblem):
    @property
    def state_observation_space(self):
        return gym.spaces.Box(-np.inf, np.inf, (4,))

    def observe_state(self, env):
        v_state = env.vehicle_state
        assert len(v_state.shape) == 2
        assert v_state.shape[1] == 7

        omega, beta = v_state[0, 5:]

        v_x, v_y = env.vehicle_model.v_loc_[0] if env.vehicle_model.v_loc_ is not None else (0, 0)

        result = np.array([
            v_x / env.vehicle_model.top_speed,
            v_y / env.vehicle_model.top_speed,
            omega,  # TODO: Good normalization?
            beta / env.vehicle_model.steering_controller.max_angle,
        ])

        return result
