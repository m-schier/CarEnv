import gymnasium as gym
import numpy as np

from .Problem import Problem
from typing import Tuple, Optional
from shapely.geometry import Point, LinearRing
from ..BatchedCones import BatchedCones
from ..Track.Generator import make_full_environment
import sys


class FreeDriveProblem(Problem):
    def __init__(self, track_width=6., cone_width=5., k_center=0., k_base=.05, k_forwards=0., extend=100.,
                 lap_limit=None, time_limit=None):
        if lap_limit is None and time_limit is None:
            raise ValueError("At least one of lap_limit and time_limit must be set")

        self.track_width = track_width
        self.cone_width = cone_width
        self.extend = extend
        self.k_center = k_center
        self.k_forwards = k_forwards
        self.k_base = k_base
        self.time_limit = time_limit
        self.lap_limit = lap_limit
        self.track_dict = None
        self.lr: Optional[LinearRing] = None
        self.old_pose_xy = None
        self.idle_time = 0.
        self.track_progress = 0.

    @property
    def state_observation_space(self):
        return gym.spaces.Box(-np.inf, np.inf, (2,))

    def observe_state(self, env):
        return np.array([
            # TODO: Normalize and not actually required
            env.vehicle_last_speed,
            env.steering_history[-1],
        ], dtype=np.float32)

    def configure_env(self, env, rng=None) -> Tuple[float, float, float]:
        self.track_dict = make_full_environment(width=self.track_width, extends=(self.extend, self.extend),
                                                cone_width=self.cone_width, rng=rng)
        self.lr = LinearRing(self.track_dict['centerline'])
        self.idle_time = 0.
        self.track_progress = 0.

        env.objects = {
            'cones': BatchedCones.from_track_dict(self.track_dict),
        }

        x, y = self.track_dict['start_xy']
        theta = self.track_dict['start_theta']
        self.old_pose_xy = np.array([x, y])
        return x, y, theta

    def calculate_forward_vel_coeff(self, env):
        pose_xy = env.ego_pose[:2]

        projection = self.lr.project(Point(pose_xy))

        track_pose = self.lr.interpolate(projection)

        next_projection = projection - .5  # Negative is forwards
        if next_projection < 0:
            next_projection += self.lr.length

        track_pose_next = self.lr.interpolate(next_projection)

        forward_dir = (np.array(track_pose_next.xy) - np.array(track_pose.xy)).flatten()
        forward_dir /= np.linalg.norm(forward_dir)  # Normalize to unit

        t_forward = np.dot(env.vehicle_model.v_, forward_dir).item()
        return t_forward

    def update(self, env, dt) -> Tuple[bool, bool]:
        env.add_to_reward(self.k_base)

        forward_v = self.calculate_forward_vel_coeff(env)
        env.metrics['forward_velocity'] = forward_v
        env.add_to_reward(forward_v * dt * self.k_forwards)

        pose_xy = env.ego_pose[:2]
        # print(f"{pose_xy = }")

        last_projection = self.lr.project(Point(self.old_pose_xy))
        new_projection = self.lr.project(Point(pose_xy))

        # Check if moving less than 0.1 m/s
        moved_distance = np.linalg.norm(pose_xy - self.old_pose_xy, axis=-1)
        if moved_distance / dt < .1:
            self.idle_time += dt
        else:
            self.idle_time = .0

        # Generators puts us backward on track, take module when wrapping
        moved = (last_projection - new_projection) % self.lr.length

        if moved > self.lr.length / 2:
            # Should be negative, moving backwards
            moved = moved - self.lr.length

        self.track_progress += moved

        # r_forward = moved * self.k_forwards
        # env.add_to_reward(r_forward)

        distance_from_center = self.lr.distance(Point(pose_xy))

        env.add_to_reward(-distance_from_center * dt * self.k_center)

        terminated = False
        truncated = False

        if distance_from_center > self.track_width / 2:
            env.set_reward(-1)
            env.add_info('Done.Reason', 'LeftTrack')
            terminated = True
        elif self.lap_limit is not None and self.track_progress > self.lr.length * self.lap_limit:
            print("Probably closed track, ending episode", file=sys.stderr)
            env.add_info('Done.Reason', 'CompletedTrack')
            truncated = True
        elif self.time_limit is not None and env.time >= self.time_limit:
            print("Time limit exceeded", file=sys.stderr)
            env.add_info('Done.Reason', 'MaxTime')
            truncated = True
        elif self.idle_time > 5.:
            print("Truncated due to idling", file=sys.stderr)
            env.add_info('Done.Reason', 'Idling')
            truncated = True

        self.old_pose_xy = pose_xy

        return terminated, truncated
