from .VelocityController import AbstractVelocityController
from .SteeringController import RateLimitedSteeringModel
import numpy as np


class BicycleModel:
    def __init__(self, velocity_controller: AbstractVelocityController,
                 wheelbase: float, beta_rate=1., beta_max=0.52):
        self.steering_model = RateLimitedSteeringModel(beta_max, beta_rate)
        self.velocity_controller = velocity_controller
        self.wheelbase = wheelbase
        self.state_ = None
        self.is_braking = None

    @property
    def n_controls(self):
        return 2

    @property
    def turning_circle(self):
        # Turning circle of the center of front axle (which is larger than the rear axle).
        return 2 * self.wheelbase / np.sin(self.steering_model.beta_max)

    def reset(self):
        self.is_braking = False
        self.steering_model.reset()
        self.state_ = np.concatenate([np.zeros(3), self.velocity_controller.initialize_state(1)[0]])

    def set_pose(self, pose):
        assert np.shape(pose) == (3,)

        self.state_[:3] = pose
        return self.state_

    def get_pose(self):
        return self.state_[:3]

    def __unpack_state(self, state: np.ndarray):
        own_state = state[:3]
        vel_state = state[3:]

        x, y, theta = own_state.T

        return x, y, theta, vel_state

    def update(self, control: np.ndarray, dt: float):
        if control.shape != (2,):
            raise ValueError(f"Bad control shape, expected {(2,)}, actual {control.shape}")

        s_control, v_control = control

        x, y, theta, v_stat = self.__unpack_state(self.state_)

        beta_old = self.steering_model.beta
        self.steering_model.integrate(s_control, dt)
        beta_new = self.steering_model.beta
        v_stat_new, v_new, v_avg = self.velocity_controller.update(v_stat, v_control, dt)
        v_stat_new, v_new, v_avg = v_stat_new[0], v_new[0], v_avg[0]

        self.is_braking = np.sign(v_control) == -np.sign(v_new)

        tan_avg = np.tan(beta_old)
        beta = np.arctan(tan_avg / 2)
        delta_x = dt * v_avg * np.cos(theta + beta)
        delta_y = dt * v_avg * np.sin(theta + beta)
        delta_theta = dt * v_avg * np.cos(beta) * tan_avg / self.wheelbase

        new_state = np.array([x + delta_x, y + delta_y, theta + delta_theta])

        result = {
            'state': np.concatenate([new_state, v_stat_new], axis=-1),
            's_new': beta_new,
            'v_new': v_new
        }

        self.state_ = np.copy(result['state'])

        return result
