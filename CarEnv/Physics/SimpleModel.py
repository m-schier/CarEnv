import numpy as np
from .SteeringController import AbstractSteeringController


def rot_mats(angles):
    assert len(angles.shape) == 1.

    cs = np.cos(angles)
    sn = np.sin(angles)

    # Faster than double np.stack
    result = np.empty((angles.shape[0], 2, 2))
    result[:, 0, 0] = cs
    result[:, 1, 1] = cs
    result[:, 0, 1] = sn
    result[:, 1, 0] = -sn

    assert result.shape == (angles.shape[0], 2, 2)
    return result


def clip_norm(vs, max_norm):
    assert len(vs.shape) == 2
    assert vs.shape[-1] == 2
    result = vs.copy()

    # Much faster than np.sum(np.square(vs), -1)...
    sq_norms = vs[:, 0] * vs[:, 0] + vs[:, 1] * vs[:, 1]
    mask = sq_norms > max_norm * max_norm
    coeffs = np.sqrt(sq_norms[mask]) / max_norm

    result[mask] /= coeffs[:, None]
    return result, mask


def soft_sign(xs, slope=10.):
    xs = xs * slope
    es = np.exp(xs)
    return (es / (es + 1.)) * 2. - 1.


def safe_slip_angle(v):
    x, y = v.T
    x = np.abs(x)
    return np.arctan2(y, np.maximum(x, .5))


class SimpleModel:
    def __init__(self, steering_controller: AbstractSteeringController, wheelbase=2.4, mass=700., inertia=300.,
                 engine_power=50000., brake_force=8000., brake_balance=.5, rwd=True, max_grip=1., stiffness_rear=6.,
                 stiffness_front=5., max_grip_opt=0.):
        self.steering_controller = steering_controller
        self.wheelbase = wheelbase
        self.mass = mass
        self.inertia = inertia  # Clearly a good number
        self.stiffness_front = -stiffness_front  # Slope of lateral force in [N/(rad*N)]
        self.stiffness_rear = -stiffness_rear
        self.max_grip = max_grip  # Maximum transferable norm of planar force per normal force of tire/axle (Kraftschlussbeiwert)
        self.max_grip_opt = max_grip_opt
        self.half_Acd0_rho = .5 * 1.25 * 1.45  # Set Acd0 to reach 70 km/h
        self.brake_force = brake_force  # Newton, total of vehicle
        self.brake_balance = brake_balance  # 1.0 is full front
        self.engine_power = engine_power  # Watt (6.6 or 21.3 for duck)
        self.C_r = .01  # Roll resistance in [N/N]
        self.rwd = rwd

        self.lat_dampen_v = None  # Speed in [m/s] at which to apply full lateral tire resistance forces for numeric stability
        self.roll_dampen_v = None  # Speed in [m/s] at which to apply full longitudinal tire resistance forces for numeric stability

        # State
        self.p_ = None
        self.theta_ = None
        self.v_ = None
        self.omega_ = None
        self.steer_state_ = None

        # Temporary variables which may be displayed
        self.v_loc_ = None
        self.v_front_ = None
        self.v_rear_ = None
        self.force_front_ = None
        self.force_rear_ = None
        self.front_slip_ = None
        self.rear_slip_ = None
        self.using_brake_ = None
        self.a_tol = 1
        # self._state = None
        # self.plot_tire_curve()

    @property
    def top_speed(self):
        """
        Terminal velocity of the vehicle in m/s.
        """
        # F_engine = P / v
        # F_roll = c
        # F_aero = a * v^2
        # P / v = a * v^2 + c
        # 0 = a * v^3 + c * v + -P
        a = self.half_Acd0_rho
        # c = 4 * 0.013 * self.mass * 9.81
        c = 0.
        d = -self.engine_power
        root = np.roots([a, 0., c, d])[-1]
        assert np.isreal(root)
        return np.real(root)

    @property
    def is_braking(self):
        return self.using_brake_ if self.using_brake_ is not None else np.zeros(len(self.p_), dtype=bool)

    def _tire_force(self, stiffness, slip_angle, long_force, downforce, v_loc=None):
        # With local velocity present, calculate forces
        assert v_loc is not None
        from .SimpleImpl import tire_forces
        return tire_forces(v_loc, long_force, downforce, slip_angle, stiffness, self.max_grip, self.C_r, self.max_grip_opt)

    def plot_tire_curve(self):
        import matplotlib.pyplot as plt
        angles = np.linspace(-90, 90, 181)
        anlges_rad = angles / 180 * np.pi
        load = self.mass * .25 * 9.81
        v_loc = np.zeros((len(angles), 2))
        v_loc[:, 1] = 10.
        ys = -self._tire_force(self.stiffness_front, anlges_rad, np.zeros_like(angles), load, v_loc=v_loc)[0][:, 1]
        plt.title(f"$F_Z$ = {load}N")
        plt.xlabel("Slip angle [°]")
        plt.ylabel("$F_Y$ [N]")
        plt.plot(angles, ys)
        plt.show()

    @property
    def peak_traction(self):
        return self.mass * 9.81 * 0.5 * self.max_grip

    def initialize_state(self, n):
        self.p_ = np.zeros((n, 2))
        self.theta_ = np.zeros(n)
        self.v_ = np.zeros((n, 2))
        self.omega_ = np.zeros(n)
        self.steer_state_ = self.steering_controller.initialize_state(n)

    def _pack_state(self):
        return np.concatenate([
            self.p_, self.theta_[:, None], self.v_, self.omega_[:, None], self.steer_state_
        ], axis=-1)

    def get_pose(self, *_):
        return np.concatenate([self.p_, self.theta_[:, None]], -1)

    def set_pose(self, pose):
        assert pose.shape == (len(self.p_), 3)
        self.p_[:] = pose[:, :2]
        self.theta_[:] = pose[:, 2]
        return self._pack_state()

    def update(self, control: np.ndarray, dt: float):
        # Read control
        assert control.shape == (len(self.p_), 3)
        steering_ctl = control[:, 0]
        acc_ctl = control[:, 1]
        brake_ctl = control[:, 2]

        # Read state
        p, v, theta, omega, steer_state = self.p_, self.v_, self.theta_, self.omega_, self.steer_state_

        # Update the steering controller, take average steering angle for time frame as beta
        new_steer_state, beta_new, beta = self.steering_controller.update(steer_state, steering_ctl, dt)

        # Convert to vehicle reference frame where x is forwards
        world_to_local = rot_mats(theta)
        local_to_world = np.transpose(world_to_local, (0, 2, 1))
        self.v_loc_ = np.squeeze(world_to_local @ v[..., None], -1)

        # Local velocity at the rear axle
        self.v_rear_ = self.v_loc_.copy()
        self.v_rear_[:, 1] -= omega * .5 * self.wheelbase

        # Local velocity at the front axle
        local_to_front = rot_mats(beta)
        front_to_local = np.transpose(local_to_front, (0, 2, 1))
        self.v_front_ = self.v_loc_.copy()
        self.v_front_[:, 1] += omega * .5 * self.wheelbase
        v_front_loc = np.squeeze(local_to_front @ self.v_front_[..., None], -1)

        slip_angle_front = safe_slip_angle(v_front_loc)
        slip_angle_rear = safe_slip_angle(self.v_rear_)

        # For indication
        self.using_brake_ = brake_ctl > 0
        # Dampen for stability of simulation
        force_brake = brake_ctl * self.brake_force * -soft_sign(self.v_loc_[:, 0])
        # Determine forward velocity of driving axle
        v_drive = self.v_rear_[:, 0] if self.rwd else v_front_loc[:, 0]
        # Calculate engine power based on drive wheel velocity
        force_engine = acc_ctl * self.engine_power / np.maximum(np.abs(v_drive), 2.)

        axle_downforce = self.mass * 9.81 * 0.5

        long_front = force_brake * self.brake_balance
        long_rear = force_brake * (1 - self.brake_balance)

        if self.rwd:
            long_rear += force_engine
        else:
            long_front += force_engine

        self.force_front_, self.front_slip_ = self._tire_force(self.stiffness_front, slip_angle_front, long_front, axle_downforce, v_loc=v_front_loc)
        self.force_front_ = np.squeeze(front_to_local @ self.force_front_[..., None], -1)

        self.force_rear_, self.rear_slip_ = self._tire_force(self.stiffness_rear, slip_angle_rear, long_rear, axle_downforce, v_loc=self.v_rear_)

        # Air resistance
        force_resistance = -self.half_Acd0_rho * self.v_loc_ * np.abs(self.v_loc_)

        # Sum of forces
        force = self.force_rear_ + self.force_front_ + force_resistance

        # Simplification: Only lateral wheel forces acting outside cog
        torque = self.force_front_[:, 1] * (self.wheelbase / 2) - self.force_rear_[:, 1] * (self.wheelbase / 2)

        acc = force / self.mass
        ang_acc = torque / self.inertia

        # Integrate

        new_v_loc = self.v_loc_ + dt * acc

        self.v_ = np.squeeze(local_to_world @ new_v_loc[..., None], -1)

        self.p_ = p + dt * self.v_
        self.omega_ = omega + dt * ang_acc
        self.theta_ = theta + dt * omega
        self.steer_state_ = new_steer_state

        return {
            'state': self._pack_state(),
            's_new': beta_new,
            'v_new': (world_to_local @ self.v_[..., None])[:, 0, 0]
        }
