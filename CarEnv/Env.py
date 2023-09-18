from typing import Optional, Tuple, Any, Dict
from enum import IntEnum
from copy import deepcopy

import gym

import numpy as np

from .Physics.BicycleModel import BicycleModel
from .Physics.SimpleModel import SimpleModel
from .BatchedObjects import BatchedObjects
from .Sensor import Sensor


class DiscreteSteerAction(IntEnum):
    LEFT = 0
    STRAIGHT = 1
    RIGHT = 2


def parse_generic(token: str):
    func_start = token.index('(')
    func_stop = token.index(')')

    assert func_stop == len(token) - 1

    func_name = token[:func_start]
    arg_start = func_start + 1
    arg_stop = func_stop

    if arg_start < arg_stop:
        args = token[arg_start:arg_stop].split(',')
    else:
        args = []

    return func_name, args


def parse_steering_model(token: str):
    from .Physics.SteeringController import LinearSteeringController, DirectSteeringController

    func_name, args = parse_generic(token)

    max_angle = 30 / 180 * np.pi

    if func_name == 'direct':
        assert len(args) == 0
        return DirectSteeringController(max_angle)
    elif func_name == 'linear':
        assert len(args) == 1
        rate = float(args[0]) / 180 * np.pi
        return LinearSteeringController(max_angle, rate)
    else:
        raise ValueError(func_name)


def make_longitudinal_model(config: dict):
    from .Physics.VelocityController import SimpleEngineDragVelocityController, DirectVelocityController, \
        LinearVelocityController
    params = config.get('longitudinal', {'type': 'linear'})

    known_types = {
        'direct': DirectVelocityController,
        'linear': LinearVelocityController,
        'simple': SimpleEngineDragVelocityController,
    }

    kwargs = {k: v for k, v in params.items() if k != 'type'}
    return known_types[params['type']](**kwargs)


def make_problem(config: dict):
    from .Problems import FreeDriveProblem, ParallelParkingProblem, RacingProblem

    params = config.get('problem', {'type': 'freedrive'})

    known_types = {
        'freedrive': FreeDriveProblem,
        'parallel_parking': ParallelParkingProblem,
        'racing': RacingProblem,
    }

    kwargs = {k: v for k, v in params.items() if k != 'type'}
    return known_types[params['type']](**kwargs)


def parse_vehicle(config: dict):
    params = config.get('vehicle', {'type': 'simple', 'wheelbase': 2.4, 'mass': 700.})

    kwargs = {k: v for k, v in params.items() if k != 'type'}
    return params['type'], kwargs


class CarEnv(gym.Env):
    metadata = {
        'render_modes': ["human", "rgb_array"]
    }

    def __init__(self, config=None, forward_range=30):
        super(CarEnv, self).__init__()

        self._rng = np.random.default_rng()
        self._config = deepcopy(config) if config is not None else {}
        self._action = self._make_action()
        self.action_space = self._action.action_space

        self.dt = config.get('dt', .2)
        self.problem = make_problem(self._config)
        self.physics_divider = config.get('physics_divider', 1)

        self.vehicle_state = None
        self.vehicle_model = None
        self.vehicle_last_speed = None
        self._reset_required = True

        self.objects: Dict[str, BatchedObjects] = {}
        self.sensors: Dict[str, Sensor] = {}
        self.metrics: Dict[str, float] = {}
        self._pending_reward = 0
        self._pending_info = {}
        self.last_observations = {}

        self.view_limits = (0, forward_range, -15, 15)
        self.collision_bb = self._config.get('collision_bb', (-1.5, 1.5, -0.8, 0.8))
        self.k_cone_hit = .2
        self.k_center = .0
        self.steering_history_length = 20

        # Rendering stuff
        self.__renderer = None
        self.__screen = None

        # Statistics
        self.steps = 0
        self.time = 0
        self.traveled_distance = 0.

        # Histories
        self.steering_history = None

        # Query sensors for obs space, do this last such that everything initialized
        obs_space = {
            "state": self.problem.state_observation_space,
        }

        for s_key, s_val in self._make_sensors().items():
            obs_space[s_key] = s_val.observation_space

        self.observation_space = gym.spaces.Dict(obs_space)

    def render(self, mode="human", width=1280, height=720):
        from .Rendering.BirdView import BirdViewRenderer

        if self.__renderer is None:
            render_hints = self.problem.render_hints if hasattr(self.problem, 'render_hints') else {}
            kwargs = {'orient_forward': render_hints.get('from_ego', True)}

            if 'scale' in render_hints:
                kwargs['scale'] = render_hints['scale']

            self.__renderer = BirdViewRenderer(width, height, **kwargs)

        rgb_array = self.__renderer.render(self)

        if mode == "rgb_array":
            return rgb_array
        elif mode == "human":
            import pygame
            if self.__screen is None:
                pygame.init()
                self.__screen = pygame.display.set_mode([width, height], flags=0, vsync=0)
            # Consume events
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    raise RuntimeError("Interrupted by user")

            # pygame expects column major
            surface = pygame.surfarray.make_surface(np.transpose(rgb_array, (1, 0, 2)))
            self.__screen.blit(surface, (0, 0))
            pygame.display.flip()
        else:
            raise ValueError(f"{mode = }")

    def seed(self, seed=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed=seed)

        # This should return the current seed in any case, but we can't retrieve a seed from numpy's default_rng
        return [seed]

    def add_to_reward(self, val):
        self._pending_reward += val

    def add_info(self, key, val):
        self._pending_info[key] = val

    def set_reward(self, val):
        self._pending_reward = val

    def step(self, action) -> Tuple[Any, float, bool, dict]:
        assert not self._reset_required

        self._pending_reward = .0
        self._pending_info = {}

        # Interpret action
        s_control, l_control = self._action.interpret(action)

        self.steps += 1
        self.time += self.dt

        control = np.concatenate([np.array([s_control]), l_control], -1)[None]
        # control = np.array([[s_control, l_control]])

        for _ in range(self.physics_divider):
            update_result = self.vehicle_model.update(control, self.dt / self.physics_divider)
            self.vehicle_state = update_result['state']

        self.steering_history = np.roll(self.steering_history, -1)
        self.steering_history[-1] = update_result['s_new'][0]
        self.vehicle_last_speed = update_result['v_new'][0]
        self.traveled_distance += self.vehicle_last_speed * self.dt

        # Update objects, check cone collision
        for obj_key, obj_val in self.objects.items():
            obj_val.update(self)

        obs = self._make_obs()

        terminated, truncated = self.problem.update(self, self.dt)

        self._reset_required = terminated or truncated
        self._pending_info['TimeLimit.truncated'] = truncated

        return obs, self._pending_reward, truncated or terminated, self._pending_info

    @property
    def action(self):
        return self._action

    @property
    def ego_pose(self):
        return self.vehicle_model.get_pose(self.vehicle_state)[0, :3]

    @property
    def ego_transform(self):
        x, y, theta = self.vehicle_model.get_pose(self.vehicle_state)[0, :3]
        c = np.cos(theta)
        s = np.sin(theta)

        R = np.array([[c, s], [-s, c]])
        t = -R @ np.array([x, y])

        trans = np.eye(3)
        trans[:2, :2] = R
        trans[:2, 2] = t

        return trans

    @property
    def view_normalizer(self):
        return max((abs(x) for x in self.view_limits))

    def _make_obs(self):
        self.last_observations = {
            'state': self.problem.observe_state(self),
        }

        for s_key, s_val in self.sensors.items():
            self.last_observations[s_key] = s_val.observe(self)

        return self.last_observations

    def _make_action(self):
        from .Actions import ContinuousSteeringAction, HumanContinuousSteeringAccelAction, ContinuousSteeringAccelAction
        from .Actions import HumanContinuousSteeringPedalsAction, ContinuousSteeringPedalsAction

        known_types = {
            'continuous_steering': ContinuousSteeringAction,
            'continuous_steering_pedals': ContinuousSteeringPedalsAction,
            'continuous_steering_accel': ContinuousSteeringAccelAction,
            'human': HumanContinuousSteeringAccelAction,
            'human_pedals': HumanContinuousSteeringPedalsAction,
        }

        action_dict = self._config.get('action', {'type': 'continuous_steering'})
        kwargs = {k: v for k, v in action_dict.items() if k != 'type'}

        return known_types[action_dict['type']](**kwargs)

    def _make_sensors(self):
        from .SensorConeMap import SensorConeMap

        known_types = {
            'conemap': SensorConeMap,
        }

        self.sensors = {}

        for s_key, s_val in self._config.get('sensors', {}).items():
            sensor_type = s_val['type']
            kwargs = {k: v for k, v in s_val.items() if k != 'type'}

            self.sensors[s_key] = known_types[sensor_type](self, **kwargs)

        return self.sensors

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None) -> Any:
        # Clean up on reset
        if self.__renderer is not None:
            self.__renderer.close()
            self.__renderer = None

        steering_controller = parse_steering_model(self._config.get('steering', 'linear(60)'))

        # velocity_controller = LinearVelocityController(130 / 3.6, 9.81, 9.81)
        veh_model, vm_kwargs = parse_vehicle(self._config)

        if veh_model == 'bicycle':
            velocity_controller = make_longitudinal_model(self._config)
            self._action.configure(steering_controller, velocity_controller)

            self.vehicle_model = BicycleModel(steering_controller, velocity_controller, **vm_kwargs)
        else:
            # TODO: Hacking in a longitudinal model which is a no-op for now
            self._action.configure(steering_controller, make_longitudinal_model({'longitudinal':{'type': 'simple'}}))
            self.vehicle_model = SimpleModel(steering_controller, **vm_kwargs)
        self.vehicle_state = self.vehicle_model.initialize_state(1)

        self.metrics = {}
        self.objects = {}

        pose = self.problem.configure_env(self, rng=self._rng)

        self._make_sensors()

        self.vehicle_state = self.vehicle_model.set_pose(np.array([pose]))
        self.vehicle_last_speed = 0.

        self._reset_required = False
        self.steps = 0
        self.time = 0.
        self.traveled_distance = 0.

        self.steering_history = np.zeros(self.steering_history_length)

        return self._make_obs()
