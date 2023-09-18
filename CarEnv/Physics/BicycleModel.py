from .SteeringController import AbstractSteeringController
from .VelocityController import AbstractVelocityController
import numpy as np


class BicycleModel:
    def __init__(self, steering_controller: AbstractSteeringController, velocity_controller: AbstractVelocityController,
                 wheelbase: float):
        self.steering_controller = steering_controller
        self.velocity_controller = velocity_controller
        self.wheelbase = wheelbase
        self.state_ = None
        self.is_braking = None

    @property
    def turning_circle(self):
        # Turning circle of the center of front axle (which is larger than the rear axle).
        return 2 * self.wheelbase / np.sin(self.steering_controller.max_angle)

    def initialize_state(self, n):
        self.is_braking = np.zeros(n, dtype=bool)
        self.state_ = np.concatenate([
            np.zeros((n, 3)),
            self.steering_controller.initialize_state(n),
            self.velocity_controller.initialize_state(n)
        ], axis=-1)

    def set_pose(self, pose):
        assert pose.shape == (len(self.state_), 3)

        self.state_[:, :3] = pose
        return self.state_

    def get_pose(self, state):
        assert len(state.shape) == 2
        return state[:, :3]

    def __unpack_state(self, state: np.ndarray):
        own_state = state[:, :3]
        steer_state = state[:, 3:3 + self.steering_controller.state_size]
        vel_state = state[:, 3 + self.steering_controller.state_size:]

        x, y, theta = own_state.T

        return x, y, theta, steer_state, vel_state

    def update(self, control: np.ndarray, dt: float):
        n_veh = len(self.state_)

        if control.shape != (n_veh, 2):
            raise ValueError(f"Bad control shape, expected {(n_veh, 2)}, actual {control.shape}")

        s_control, v_control = control.T

        x, y, theta, s_stat, v_stat = self.__unpack_state(self.state_)

        s_stat_new, s_new, s_avg = self.steering_controller.update(s_stat, s_control, dt)
        assert s_stat_new.shape == (n_veh, self.steering_controller.state_size)
        assert s_new.shape == (n_veh,)
        assert s_avg.shape == (n_veh,)
        v_stat_new, v_new, v_avg = self.velocity_controller.update(v_stat, v_control, dt)
        assert v_stat_new.shape == (n_veh, self.velocity_controller.state_size)
        assert v_new.shape == (n_veh,)
        assert v_avg.shape == (n_veh,)

        self.is_braking = np.sign(v_control) == -np.sign(v_new)

        tan_avg = np.tan(s_avg)
        beta = np.arctan(tan_avg / 2)
        delta_x = dt * v_avg * np.cos(theta + beta)
        delta_y = dt * v_avg * np.sin(theta + beta)
        delta_theta = dt * v_avg * np.cos(beta) * tan_avg / self.wheelbase

        new_state = np.stack([x + delta_x, y + delta_y, theta + delta_theta], axis=-1)

        result = {
            'state': np.concatenate([new_state, s_stat_new, v_stat_new], axis=-1),
            's_new': s_new,
            'v_new': v_new
        }

        self.state_ = np.copy(result['state'])

        return result
