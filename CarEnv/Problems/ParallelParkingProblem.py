from typing import Tuple

import gymnasium as gym
import numpy as np

from .Problem import Problem
from ..BatchedCones import BatchedCones


def angle_difference(alpha, beta):
    """
    Return difference of radian angles alpha and beta in radian in range (-pi, pi]
    """

    diff = (alpha - beta) % (2 * np.pi)
    assert diff >= 0

    if diff > np.pi:
        return diff - 2 * np.pi
    else:
        return diff


class ParallelParkingProblem(Problem):
    def __init__(self, start="before", k_continuous=.05, max_time=None):
        self.target_pose = None
        self.k_continuous = k_continuous

        if max_time is None:
            self.max_time = 15. if start == 'before' else 7.
        else:
            self.max_time = max_time

        self.start = start
        self.help_pos = None

    @property
    def render_hints(self) -> dict:
        return {
            'scale': 30,
            'from_ego': False,
        }

    @property
    def state_observation_space(self):
        return gym.spaces.Box(-1, 1, (2,))

    def observe_state(self, env):
        return np.array([
            # TODO: Better normalization
            env.vehicle_last_speed / 10.,
            env.steering_history[-1],
        ])

    def _make_problem(self, env, rng=None):
        gap_start = rng.integers(5, 10) * 2
        gap_size = 8
        gap_stop = gap_start + gap_size
        gap_width = 2
        street_width = 5
        car_width = env.collision_bb[3] - env.collision_bb[2]

        start_x = 0. if self.start == 'before' else 20 + gap_size + 5 + rng.uniform()

        # 2m spacing on longitudinal
        pos0 = np.stack([
            np.linspace(0, 50, 26),
            np.ones(26) * -street_width,
        ], axis=-1)
        pos1 = np.stack([
            np.linspace(0, gap_start, gap_start // 2 + 1),
            np.zeros(gap_start // 2 + 1),
        ], axis=-1)

        # 1m spacing on lateral
        pos2 = np.stack([
            np.ones(gap_width - 1) * gap_start,
            np.linspace(1, gap_width - 1, gap_width - 1),
        ], axis=-1)

        # 2m spacing on longitudinal
        pos3 = np.stack([
            np.linspace(gap_start, gap_start + gap_size, gap_size // 2 + 1),
            np.ones(gap_size // 2 + 1) * gap_width
        ], axis=-1)

        # 1m spacing on lateral
        pos4 = np.stack([
            np.ones(gap_width - 1) * gap_stop,
            np.linspace(1, gap_width - 1, gap_width - 1),
        ], axis=-1)

        # 2m spacing on longitudinal
        pos5 = np.stack([
            np.linspace(gap_stop, 50, (50 - gap_stop) // 2 + 1),
            np.zeros((50 - gap_stop) // 2 + 1),
        ], axis=-1)

        # Zero mean on x
        cones_pos = np.concatenate([pos0, pos1, pos2, pos3, pos4, pos5]) + np.array([-25, street_width / 2])

        # Random angle [-10°, 10°)
        start_angle = (rng.uniform() - .5) * 20 / 180 * np.pi

        start_pose = start_x - 25, rng.uniform() - .5, start_angle

        target_pose = np.array([gap_start + gap_size / 2 - 25, street_width / 2 + car_width / 2, 0.])

        return start_pose, target_pose, cones_pos

    def configure_env(self, env, rng=None) -> Tuple[float, float, float]:
        start_pose, target_pose, cones_pos = self._make_problem(env, rng)

        self.target_pose = target_pose

        cones = np.concatenate([
            cones_pos,
            np.ones_like(cones_pos[:, :1]),
            np.zeros_like(cones_pos[:, :1]),
        ], axis=-1)

        env.objects = {
            'cones': BatchedCones(cones)
        }

        return start_pose

    def render(self, ctx, env):
        import cairo
        from ..Rendering.Rendering import stroke_fill
        ctx: cairo.Context = ctx

        ctx.arc(*self.target_pose[:2], .1, 0, 2 * np.pi)
        ctx.close_path()
        stroke_fill(ctx, (0., 0., 0.), None)

        veh_pose = env.ego_pose
        ctx.arc(*veh_pose[:2], .075, 0, 2 * np.pi)
        ctx.close_path()
        stroke_fill(ctx, (0., 0., 0.), None)

    def pose_reward(self, current_pose, target_pose):
        current_xy, current_alpha, target_xy, target_alpha = current_pose[:2], current_pose[2], target_pose[:2], target_pose[2]
        pos_diff = np.abs(np.array(current_xy) - np.array(target_xy))

        reward_x = np.maximum(0., 20 - pos_diff[0]) / 20
        reward_y = np.maximum(0., 5. - pos_diff[1]) / 5
        reward_angle = np.maximum(0., np.cos(current_alpha - target_alpha))

        return reward_x * reward_y * reward_angle

    def is_out_of_bounds(self, pose):
        return pose[0] < -25 or pose[0] > 25

    def update(self, env, dt: float) -> Tuple[bool, bool]:
        if env.objects['cones'].hit_count > 0:
            env.set_reward(-1)
            return True, False

        pose = env.ego_pose

        if self.is_out_of_bounds(pose):
            env.set_reward(-1)
            return True, False

        reward_combined = self.pose_reward(pose, self.target_pose)

        env.add_to_reward(reward_combined * self.k_continuous)

        truncated = env.time >= self.max_time

        return False, truncated
