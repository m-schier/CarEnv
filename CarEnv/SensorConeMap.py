import cairo
import gymnasium as gym
import numpy as np

from .Sensor import Sensor
from .BatchedCones import BatchedCones
from .Util import transform_from_pose


class SensorConeMap(Sensor):
    def __init__(self, env, proxy, max_objects=100, normalize=True, bbox=(-30, 30, -30, 30), limit_type=None,
                 noise_profile=None, sort=False):
        super(SensorConeMap, self).__init__(env, proxy)
        self._max_objects = max_objects
        self._normalize = normalize
        self._bbox = bbox
        self.limit_type = limit_type
        self.sort = sort
        self.imperfection_misclassify_chance = 0.

        if isinstance(noise_profile, float):
            self.imperfection_miss_chance = noise_profile * .1
            self.imperfection_position_error_scale = noise_profile / 30.
        elif noise_profile is None:
            self.imperfection_miss_chance = 0.
            self.imperfection_position_error_scale = 0.
        elif noise_profile == 'low':
            self.imperfection_miss_chance = 0.05
            self.imperfection_position_error_scale = 0.5 / 30.
        elif noise_profile == 'high':
            self.imperfection_miss_chance = 0.15
            self.imperfection_position_error_scale = 1.5 / 30.
        else:
            raise ValueError(f"{noise_profile = }")

    @property
    def bbox(self):
        return tuple(self._bbox)

    def draw(self, ctx: cairo.Context):
        from .Rendering.Rendering import stroke_fill
        from .Rendering.Colors import CONE_BLUE, CONE_YELLOW, BACKGROUND

        last_obs = self.last_observation

        if last_obs is None:
            return

        if self.limit_type is not None:
            from warnings import warn
            warn(f"SensorConeMap: Drawing not supported if limiting type")
            return

        draw_size = 250.
        ctx.save()
        ctx.identity_matrix()
        ctx.rectangle(0., 0., draw_size, draw_size)
        ctx.clip_preserve()
        stroke_fill(ctx, (0., 0., 0.), BACKGROUND, line_width=4.)
        ctx.translate(draw_size / 2, draw_size / 2)
        ctx.scale(draw_size / 2, draw_size / 2)

        for present, x, y, is_blue, is_yellow in last_obs:
            if present <= 0:
                continue

            if not self._normalize:
                x, y = x / self.view_normalizer, y / self.view_normalizer

            fill_color = CONE_BLUE if is_blue > 0 else CONE_YELLOW

            ctx.arc(y, -x, .04, 0., 2 * np.pi)
            stroke_fill(ctx, (0., 0., 0.), fill_color)

        ctx.restore()
        ctx.reset_clip()

    @property
    def observation_space(self) -> gym.Space:
        if self.limit_type is None:
            return gym.spaces.Box(-np.inf, np.inf, shape=(self._max_objects, 1 + 4))
        else:
            return gym.spaces.Box(-np.inf, np.inf, shape=(self._max_objects, 1 + 2))

    @property
    def view_normalizer(self):
        return max((abs(x) for x in self._bbox))

    def _observe_impl(self):
        env = self.env

        cones: BatchedCones = env.objects['cones']
        ego_pose = self.proxy.model.pose
        transform = transform_from_pose(ego_pose)

        if self.limit_type is not None:
            cones = cones.filtered_by_type(self.limit_type)

        cones = cones.transformed(transform).filtered_aabb(*self._bbox)

        vis_pos = cones.data[:, :2]
        cone_types_idx = np.argmax(cones.data[:, 2:], axis=-1) + 1

        # Imperfection and noise in observation model
        if self.imperfection_position_error_scale > 0.:
            cone_pos_offset = np.random.normal(loc=0, scale=self.imperfection_position_error_scale, size=vis_pos.shape)
            # cone_pos_offset = np.clip(cone_pos_offset, -self.imperfection_position_error_clip, self.imperfection_position_error_clip)
            vis_pos = vis_pos + cone_pos_offset * np.linalg.norm(vis_pos, axis=-1)[:, None]

        if self.imperfection_miss_chance > 0.:
            mask = np.random.random(len(cones)) >= self.imperfection_miss_chance
            vis_pos = vis_pos[mask]
            cone_types_idx = cone_types_idx[mask]

        if self.imperfection_misclassify_chance > .0:
            misclassified = np.random.random(len(vis_pos)) < self.imperfection_misclassify_chance
            # # TODO: Must be updated if cone_types modified
            cone_types_idx[misclassified] = 3 - cone_types_idx[misclassified]

        # Normalization
        if self._normalize:
            vis_pos = vis_pos / self.view_normalizer

        count = vis_pos.shape[0]
        enc_count = min(count, self._max_objects)

        if enc_count < count:
            import warnings
            warnings.warn(f"Discarding {count - enc_count} objects because {self._max_objects = } is too low.")

        if self.limit_type is None:
            result = np.zeros((self._max_objects, cones.data.shape[1] + 1), dtype=np.float32)
            result[:enc_count, 0] = 1
            result[:enc_count, 1:3] = vis_pos[:enc_count]
            result[:enc_count, 3:] = BatchedCones.categorical_from_indices(cone_types_idx[:enc_count])
        else:
            result = np.zeros((self._max_objects, 3), dtype=np.float32)
            result[:enc_count, 0] = 1
            result[:enc_count, 1:3] = vis_pos[:enc_count]

        if self.sort:
            order = np.argsort(np.linalg.norm(vis_pos[:enc_count], axis=-1))
            result[:enc_count] = result[order]

        return result
