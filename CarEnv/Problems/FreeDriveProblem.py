import cairo
import gymnasium as gym
import numpy as np

from .Problem import Problem
from typing import Tuple, Optional
from shapely.geometry import Point, LinearRing
from ..BatchedCones import BatchedCones
from ..Track.Generator import make_full_environment


class DirLookup:
    """
    A very fast approximation of finding the forward direction along a LinearRing for a given interpolation.
    Avoids using shapely after initialization.
    """

    def __init__(self, lr: LinearRing):
        # Construct segment lengths and offsets from LinearRing
        coords = np.asarray(lr.coords)
        offsets = np.roll(coords, -1, axis=0) - coords
        lengths = np.linalg.norm(offsets, axis=-1)

        # May have zero length segments if following points equal, filter out
        mask = lengths > 0.
        offsets = offsets[mask]
        lengths = lengths[mask]

        # Calculate normalized directions
        self._cum_lengths = np.cumsum(lengths)
        self._dirs = offsets / lengths[:, None]

    def __call__(self, interp):
        # Could interpolate here, but just return closest
        idx = np.searchsorted(self._cum_lengths, interp)
        return self._dirs[idx]


def generate_standard_track(rng, track_width=6., cone_width=5., extend=100.):
    return make_full_environment(width=track_width, extends=(extend, extend), cone_width=cone_width, rng=rng)


class FreeDriveProblem(Problem):
    def __init__(self, env, track_width=6., cone_width=5., k_center=0., k_base=.05, k_forwards=0., extend=100.,
                 lap_limit=None, time_limit=None, soft_collisions=False, truncate_idle=True, place_cones=True):
        from functools import partial
        super(FreeDriveProblem, self).__init__(env)

        if lap_limit is None and time_limit is None:
            raise ValueError("At least one of lap_limit and time_limit must be set")

        self.k_center = k_center
        self.k_forwards = k_forwards
        self.k_base = k_base
        self.time_limit = time_limit
        self.lap_limit = lap_limit
        self.soft_collisions = soft_collisions
        self.truncate_idle = truncate_idle
        self.place_cones = place_cones
        self.track_dict = None
        self.lr: Optional[LinearRing] = None
        self.old_pose_xy = None
        self.old_projection = None
        self.idle_time = 0.
        self.track_progress = 0.
        self.generate_track = partial(generate_standard_track, track_width=track_width, cone_width=cone_width, extend=extend)
        self._dir_lookup = None
        self._track_renderer = None

    @property
    def state_observation_space(self):
        return gym.spaces.Box(-np.inf, np.inf, (2,))

    def observe_state(self):
        return np.array([
            # TODO: Normalize and not actually required
            self.env.vehicle_model.velocity[0] / 10.,
            self.env.vehicle_model.steering_angle,
        ], dtype=np.float32)

    def configure_env(self, rng=None) -> Tuple[float, float, float]:
        from ..Track.Util import shapely_safe_buffer
        from ..Metrics import EpisodeAveragingMetric

        # Reset state
        self.idle_time = 0.
        self.track_progress = 0.
        self._track_renderer = None

        # Configure new state
        self.track_dict = self.generate_track(rng)
        self.lr = LinearRing(self.track_dict['centerline'])
        # Add outline to track
        self.track_dict['poly'] = shapely_safe_buffer(self.lr, self.track_dict['width'] / 2 + .3)
        self._dir_lookup = DirLookup(self.lr)

        if self.place_cones:
            self.env.add_object('cones', BatchedCones.from_track_dict(self.track_dict,
                                                                      soft_collision_distance=.3 if self.soft_collisions
                                                                      else None))

        self.env.metrics['AvgTrackVelocity'] = EpisodeAveragingMetric()

        x, y = self.track_dict['start_xy']
        theta = self.track_dict['start_theta']
        self.old_pose_xy = np.array([x, y])
        self.old_projection = self.lr.project(Point(self.old_pose_xy))
        return x, y, theta

    def calculate_forward_vel_coeff(self, env, projection=None):
        pose_xy = env.ego_pose[:2]

        if projection is None:
            projection = self.lr.project(Point(pose_xy))

        forward_dir = self._dir_lookup(projection)
        t_forward = np.dot(env.vehicle_model.v_, forward_dir).item()
        return t_forward

    def trajectory_for_proxy(self, proxy):
        """
        Return the trajectory for the given proxy for compatibility with the trajectory sensor
        :param proxy: Vehicle proxy, must be the ego vehicle
        :return: Trajectory as LinearRing
        """
        if proxy != self.env.agent.proxy:
            raise KeyError

        return LinearRing(self.track_dict['centerline'][::-1])

    def update(self, dt) -> Tuple[bool, bool]:
        env = self.env

        env.add_to_reward(self.k_base)

        pose_xy = env.ego_pose[:2]
        new_projection = self.lr.project(Point(pose_xy))

        # Forward is negative along LinearRing
        forward_v = -self.calculate_forward_vel_coeff(env, projection=new_projection)
        env.add_to_reward(forward_v * dt * self.k_forwards)

        # Check if moving less than 0.1 m/s
        moved_distance = np.linalg.norm(pose_xy - self.old_pose_xy, axis=-1)
        if moved_distance / dt < .1:
            self.idle_time += dt
        else:
            self.idle_time = .0

        # Generators puts us backward on track, take module when wrapping
        moved = (self.old_projection - new_projection) % self.lr.length

        if moved > self.lr.length / 2:
            # Should be negative, moving backwards
            moved = moved - self.lr.length

        env.metrics['AvgTrackVelocity'].update(moved / dt)

        self.track_progress += moved

        on_point = self.lr.interpolate(new_projection)
        distance_from_center = self.lr.distance(Point(pose_xy))
        env.add_to_reward(-distance_from_center * dt * self.k_center)

        terminated = False
        truncated = False

        if distance_from_center > self.track_dict['width'] / 2:
            env.set_reward(-1)
            env.add_info('Done.Reason', 'LeftTrack')
            terminated = True
        elif self.lap_limit is not None and self.track_progress > self.lr.length * self.lap_limit:
            env.add_info('Done.Reason', 'CompletedTrack')
            truncated = True
        elif self.time_limit is not None and env.time >= self.time_limit:
            env.add_info('Done.Reason', 'MaxTime')
            truncated = True
        elif self.idle_time > 5. and self.truncate_idle:
            env.add_info('Done.Reason', 'Idling')
            truncated = True

        self.old_pose_xy = pose_xy
        self.old_projection = new_projection

        return terminated, truncated

    def draw(self, draw):
        from ..Rendering.TrackRenderer import TrackRenderer
        from ..Rendering.DrawLevels import LEVEL_GROUND

        if self._track_renderer is None:
            self._track_renderer = TrackRenderer()

        draw.defer(LEVEL_GROUND, self._deferred_draw)

    def _deferred_draw(self, ctx: cairo.Context):
        self._track_renderer.render(ctx, self.track_dict['centerline'], self.track_dict['poly'])


