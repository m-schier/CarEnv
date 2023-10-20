import gymnasium as gym
import numpy as np
from .FreeDriveProblem import FreeDriveProblem


class RacingProblem(FreeDriveProblem):
    @property
    def state_observation_space(self):
        return gym.spaces.Box(-np.inf, np.inf, (6,))

    def observe_state(self, env):
        # v_state = env.vehicle_state
        # assert len(v_state.shape) == 2
        # assert v_state.shape[1] >= 7
        # omega, beta = v_state[0, 5:7]
        omega, beta = env.vehicle_model.omega_, env.vehicle_model.beta_

        v_x, v_y = env.vehicle_model.v_loc_ if env.vehicle_model.v_loc_ is not None else (0, 0)

        if hasattr(env.vehicle_model, 'omega_front_'):
            omega_front, omega_rear = env.vehicle_model.omega_front_, env.vehicle_model.omega_rear_
            omega_front /= env.vehicle_model.max_angular_velocity
            omega_rear /= env.vehicle_model.max_angular_velocity
        else:
            omega_front, omega_rear = 0., 0.

        result = np.array([
            v_x / env.vehicle_model.top_speed,
            v_y / env.vehicle_model.top_speed,
            omega,  # TODO: Good normalization?
            beta / env.vehicle_model.steering_model.beta_max,
            omega_front,
            omega_rear,
        ], dtype=np.float32)

        return result
