import numpy as np
from .SteeringController import RateLimitedSteeringModel
from numba import jit


@jit(nopython=True)
def _tire_forces(c_sigma, c_alpha, sigma_x, v_loc, mu, F_z):
    F_tilde_x = c_sigma * (sigma_x / (1 + np.abs(sigma_x))) * F_z
    # Safe tan
    tan_alpha = v_loc[1] / (np.abs(v_loc[0]) + 1e-3)
    # Dampening to prevent oscillation while barely moving
    # TODO: Necessary?
    # tan_alpha = tan_alpha * np.abs(soft_sign(v_loc[:, 0]))
    F_tilde_y = -c_alpha * (tan_alpha / (1 + np.abs(sigma_x))) * F_z
    lambda_ = mu * F_z / (2 * np.sqrt(F_tilde_x ** 2 + F_tilde_y ** 2) + 1e-10)
    f_lambda = (2. - lambda_) * lambda_ if lambda_ <= 1. else 1.
    # Boundary at which to identify axle as slipping, threshold somewhat arbitrary but should be less than 1.
    slipping = lambda_ <= .3
    return np.array([
        F_tilde_x * f_lambda,
        F_tilde_y * f_lambda
    ]), slipping


@jit(nopython=True)
def _rotate(vec2d, angle):
    x, y = vec2d
    cs = np.cos(angle)
    sn = np.sin(angle)
    return np.array([cs * x + sn * y, -sn * x + cs * y])


@jit(nopython=True)
def _calc_velocities(v_global, theta, omega, wheelbase):
    v_loc = _rotate(v_global, theta)
    v_rear = np.array([v_loc[0], v_loc[1] - omega * .5 * wheelbase])
    v_front = np.array([v_loc[0], v_loc[1] + omega * .5 * wheelbase])
    return v_loc, v_front, v_rear


@jit(nopython=True)
def _clip(v, v_min, v_max):
    if v < v_min:
        return v_min
    elif v > v_max:
        return v_max
    else:
        return v


@jit(nopython=True)
def _integrate(dt, pos, v_global, omega, omega_front, omega_rear, theta, beta, acc_ctl, brake_ctl, mass, inertia, wheelbase, inertia_front, inertia_rear, half_Acd0_rho, C_r, r_eff, engine_torque, engine_power, brake_torque, brake_balance, rwd, c_alpha_front, c_sigma_front, c_alpha_rear, c_sigma_rear, mu_front, mu_rear):
    # Convert to vehicle reference frame where x is forwards
    v_loc, v_front, v_rear = _calc_velocities(v_global, theta, omega, wheelbase)
    v_front_loc = _rotate(v_front, beta)

    # Only for indication, small dead zone for analog inputs
    using_brake_ = brake_ctl > .05

    # Roll resistance
    F_z_axle = mass * 9.81 * 0.5
    M_roll = C_r * F_z_axle * r_eff

    # Maximum available brake moments (includes rolling)
    M_brake_front_av = brake_ctl * brake_torque * brake_balance + M_roll
    M_brake_rear_av = brake_ctl * brake_torque * (1. - brake_balance) + M_roll

    # Moment from engine
    omega_drive = omega_rear if rwd else omega_front
    M_max = engine_torque * acc_ctl
    torque_engine_ = _clip(engine_power / (np.abs(omega_drive) + 1e-5) * acc_ctl, -M_max, M_max)

    # Tire forces
    sigma_x_front_ = (omega_front * r_eff - v_front_loc[0]) / (
                np.maximum(np.abs(omega_front * r_eff), np.abs(v_front_loc[0])) + 1e-10)
    sigma_x_rear_ = (omega_rear * r_eff - v_rear[0]) / (
                np.maximum(np.abs(omega_rear * r_eff), np.abs(v_rear[0])) + 1e-10)
    force_front_, front_slip_ = _tire_forces(c_sigma_front, c_alpha_front, sigma_x_front_,
                                                       v_front_loc, mu_front, F_z_axle)
    force_rear_, rear_slip_ = _tire_forces(c_sigma_rear, c_alpha_rear, sigma_x_rear_,
                                                     v_rear, mu_rear, F_z_axle)
    M_tire_front = -force_front_[0] * r_eff
    M_tire_rear = -force_rear_[0] * r_eff

    # Total moments on axles
    M_total_front = M_tire_front
    M_total_rear = M_tire_rear

    if not rwd:
        M_total_front += torque_engine_
    else:
        M_total_rear += torque_engine_

    # Limit actual brake moments from available to prevent oscillation in euler integration if wheels stopped
    M_brake_front_used = np.where(omega_front == 0.,
                                  _clip(-M_total_front, -M_brake_front_av, M_brake_front_av),
                                  M_brake_front_av * -np.sign(omega_front))
    M_brake_rear_used = np.where(omega_rear == 0.,
                                 _clip(-M_total_rear, -M_brake_rear_av, M_brake_rear_av),
                                 M_brake_rear_av * -np.sign(omega_rear))

    # Determine if braking torque exceeded all other forms of torque (and in opposite direction)
    front_brake_dominant = np.logical_and(np.abs(M_brake_front_used) > np.abs(M_total_front),
                                          np.sign(M_brake_front_used) != np.sign(M_total_front))
    rear_brake_dominant = np.logical_and(np.abs(M_brake_rear_used) > np.abs(M_total_rear),
                                         np.sign(M_brake_rear_used) != np.sign(M_total_rear))

    # Add brake moments to total moments now
    M_total_front = M_total_front + M_brake_front_used
    M_total_rear = M_total_rear + M_brake_rear_used

    force_front_ = _rotate(force_front_, -beta)

    # Air resistance
    force_resistance = -half_Acd0_rho * v_loc * np.abs(v_loc)

    # Sum of forces
    F_total = force_front_ + force_rear_ + force_resistance

    # Simplification: Only lateral wheel forces acting outside cog
    torque = force_front_[1] * (wheelbase / 2) - force_rear_[1] * (wheelbase / 2)

    acc_total = F_total / mass
    ang_acc = torque / inertia

    d_omega_front = M_total_front / inertia_front
    d_omega_rear = M_total_rear / inertia_rear

    # Integrate
    new_v_loc = v_loc + dt * acc_total
    new_omega_front = omega_front + dt * d_omega_front
    new_omega_rear = omega_rear + dt * d_omega_rear
    new_omega = omega + dt * ang_acc
    new_theta = theta + dt * new_omega
    new_v_global = _rotate(new_v_loc, -theta)

    # Prevent zero crossing of axle angular velocities if brake dominant (Coulomb friction)
    if np.sign(new_omega_front) != np.sign(omega_front) and front_brake_dominant:
        new_omega_front = 0.

    if np.sign(new_omega_rear) != np.sign(omega_rear) and rear_brake_dominant:
        new_omega_rear = 0.

    # Prevent oscillation around 0 when nearly standing with euler integration
    veh_nearly_stationary = np.logical_and((v_global[0] ** 2 + v_global[1] ** 2) < .01, np.abs(new_omega) < .05)
    wheels_stationary = np.logical_and(np.abs(new_omega_front) < 1e-5, np.abs(new_omega_rear) < 1e-5)

    if veh_nearly_stationary and wheels_stationary:
        new_omega = 0.
        new_v_global = np.zeros(2)

    # Continue integrating
    new_pos = pos + dt * new_v_global
    new_omega = new_omega
    new_theta = new_theta
    new_omega_front = new_omega_front
    new_omega_rear = new_omega_rear
    new_v_loc, new_v_front, new_v_rear = _calc_velocities(new_v_global, new_theta, new_omega, wheelbase)

    # self.p_, self.v_, self.v_loc, self.v_front_, self.v_rear_, self.omega_, self.omega_front_, self.omega_rear_

    return new_pos, new_theta, new_v_global, new_v_loc, new_v_front, new_v_rear, new_omega, new_omega_front, \
           new_omega_rear, force_front_, front_slip_, force_rear_, rear_slip_, sigma_x_front_, sigma_x_rear_, \
           using_brake_


class SingleTrackDugoffModel:
    def __init__(self, wheelbase=2.4, mass=700., inertia=300.,
                 engine_power=50000., engine_torque=400., brake_torque=8000., brake_balance=.5, rwd=True,
                 c_alpha_rear=6., c_alpha_front=5., c_sigma_rear=6., c_sigma_front=5., mu_front=1., mu_rear=1.,
                 r_eff=.3, inertia_front=1., inertia_rear=1., C_r=.015, A_Cd_rho=0.86, beta_rate=1., beta_max=0.52):
        self.steering_model = RateLimitedSteeringModel(beta_max, beta_rate)
        self.wheelbase = wheelbase
        self.mass = mass
        self.inertia = inertia  # Clearly a good number
        self.c_alpha_rear = c_alpha_rear
        self.c_alpha_front = c_alpha_front
        self.c_sigma_rear = c_sigma_rear
        self.c_sigma_front = c_sigma_front
        self.mu_front = mu_front
        self.mu_rear = mu_rear
        self.half_Acd0_rho = .5 * A_Cd_rho
        self.brake_torque = brake_torque
        self.engine_torque = engine_torque
        self.brake_balance = brake_balance  # 1.0 is full front
        self.engine_power = engine_power  # Watt
        self.C_r = C_r
        self.r_eff = r_eff
        self.inertia_front = inertia_front
        self.inertia_rear = inertia_rear
        self.rwd = rwd

        self.lat_dampen_v = None  # Speed in [m/s] at which to apply full lateral tire resistance forces for numeric stability
        self.roll_dampen_v = None  # Speed in [m/s] at which to apply full longitudinal tire resistance forces for numeric stability

        # State
        self.p_ = None
        self.theta_ = None
        self.v_ = None
        self.omega_ = None
        self.omega_front_ = None
        self.omega_rear_ = None

        # Temporary variables which may be displayed
        self.v_loc_ = None
        self.v_front_ = None
        self.v_rear_ = None
        self.torque_engine_ = None
        self.sigma_x_front_ = None
        self.sigma_x_rear_ = None
        self.force_front_ = None
        self.force_rear_ = None
        self.front_slip_ = None
        self.rear_slip_ = None
        self.using_brake_ = None
        self.a_tol = 1

    @property
    def beta_(self):
        return self.steering_model.beta

    @property
    def n_controls(self):
        return 3

    @property
    def peak_traction(self):
        return max(self.mu_rear, self.mu_front) * self.mass * 9.81 * 0.5

    @property
    def max_angular_velocity(self):
        return self.top_speed / self.r_eff

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
        return self.using_brake_ if self.using_brake_ is not None else False

    def reset(self):
        self.p_ = np.zeros(2)
        self.theta_ = 0.
        self.v_ = np.zeros(2)
        self.omega_ = 0.
        self.omega_front_ = 0.
        self.omega_rear_ = 0.
        self._update_velocities()
        self.steering_model.reset()

    def get_pose(self):
        return np.concatenate([self.p_, [self.theta_]])

    def set_pose(self, pose):
        assert np.shape(pose) == (3,)
        self.p_[:] = pose[:2]
        self.theta_ = pose[2]

    def _update_velocities(self):
        self.v_loc_, self.v_front_, self.v_rear_ = _calc_velocities(self.v_, self.theta_, self.omega_, self.wheelbase)

    def update(self, control: np.ndarray, dt: float):
        # Read control
        assert control.shape == (3,)
        steering_ctl, acc_ctl, brake_ctl = control

        # Update the steering controller, take average steering angle for time frame as beta
        beta = self.steering_model.beta
        self.steering_model.integrate(steering_ctl, dt)
        beta_new = self.steering_model.beta

        s = _integrate(dt, self.p_, self.v_, self.omega_, self.omega_front_, self.omega_rear_, self.theta_, beta, acc_ctl,
                       brake_ctl, self.mass, self.inertia, self.wheelbase, self.inertia_front, self.inertia_rear,
                       self.half_Acd0_rho, self.C_r, self.r_eff, self.engine_torque, self.engine_power,
                       self.brake_torque, self.brake_balance, self.rwd, self.c_alpha_front, self.c_sigma_front,
                       self.c_alpha_rear, self.c_sigma_rear, self.mu_front, self.mu_rear)

        self.p_, self.theta_, self.v_, self.v_loc_, self.v_front_, self.v_rear_, self.omega_, self.omega_front_, \
        self.omega_rear_, self.force_front_, self.front_slip_, self.force_rear_, self.rear_slip_, self.sigma_x_front_,\
        self.sigma_x_rear_, self.using_brake_ = s

        return {
            'state': None,
            's_new': beta_new,
            'v_new': self.v_loc_[0],
        }
