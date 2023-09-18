import numpy as np
from typing import Tuple


class AbstractVelocityController:
    @property
    def state_size(self) -> int:
        raise NotImplementedError

    def initialize_state(self, n: int):
        return np.zeros((n, self.state_size))

    def update(self, state: np.ndarray, control: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Update state
        :return: Tuple of [New state, Final velocity, Average velocity]
        """
        raise NotImplementedError


class DirectVelocityController(AbstractVelocityController):
    def __init__(self, max_v):
        self.max_v = max_v

    @property
    def state_size(self) -> int:
        return 0

    def update(self, state: np.ndarray, control: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return np.empty((len(control), 0)), np.clip(control, 0, self.max_v), np.clip(control, 0, self.max_v)


class SimpleEngineDragVelocityController(AbstractVelocityController):
    def __init__(self, engine_power: float = 20., Acd0: float = 3., mass: float = 800., brake_force=12000.):
        """

        Args:
            engine_power: Engine power in kW
            Acd0: Parasite drag area
            mass: Mass in kg
            brake_force: Braking force in newtons
        """
        self.engine_power = engine_power * 1000.  # kW to W
        self.Acd0 = Acd0
        self.mass = mass
        self.brake_force = brake_force

    @property
    def state_size(self) -> int:
        return 1  # Velocity

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
        a = 1.25 / 2 * self.Acd0
        c = 4 * 0.013 * self.mass * 9.81
        d = -self.engine_power
        root = np.roots([a, 0., c, d])[-1]
        assert np.isreal(root)
        return np.real(root)

    def update(self, state: np.ndarray, control: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        state = state.flatten()

        # Small control deadzone
        control = np.where(np.abs(control) < .05, np.zeros_like(control), control)

        # F_W = rho / 2 * c_w * A * v^2
        # rho = 1.25
        force_parasite = 1.25 / 2 * self.Acd0 * np.square(state) * -np.sign(state)

        # F_R = c_R * F_N = c_R * m * g
        # c_R = 0.013 (car tire on asphalt)
        force_rolling = 4 * 0.013 * self.mass * 9.81 * -np.sign(state)

        # P = F * v <=> F = P / v
        # Hack static power
        using_brake = np.sign(control) == -np.sign(state)
        force_brake = control * self.brake_force
        force_engine = control * self.engine_power / np.maximum(np.abs(state), 2.)
        force_control = np.where(using_brake, force_brake, force_engine)
        # force_engine = requested_power / state

        # print(f"{force_engine = }, {force_rolling = }, {force_parasite = }")

        # F = m * a <=> a = F / m
        accel = (force_control + force_rolling + force_parasite) / self.mass

        # print(f"{accel = }")

        new_state = state + accel * dt

        # print(f"{new_state = }, {state = }")

        # Fix to zero on zero crossing (otherwise would accelerate using braking force and vice-versa)
        new_state = np.where(
            np.sign(new_state) == -np.sign(state),
            np.zeros_like(new_state),
            new_state
        )

        return new_state[:, None], new_state, (new_state + state) / 2


class LinearVelocityController(AbstractVelocityController):
    def __init__(self, max_v: float = 130 / 3.6, accel: float = 9.81, decel: float = 9.81):
        self.max_v = max_v
        self.accel = accel
        self.decel = decel

    @property
    def state_size(self) -> int:
        return 1

    def update(self, state: np.ndarray, control: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        state = state.flatten()
        # Limit control to max accel and decel
        v_new = np.clip(control, state - self.decel * dt, state + self.accel * dt)
        # Also limit to no speed and max speed
        v_new = np.clip(v_new, 0, self.max_v)
        return v_new[:, None], v_new, (v_new + state) / 2
