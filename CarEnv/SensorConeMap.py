import gym
import numpy as np

from .Sensor import Sensor
from .BatchedCones import BatchedCones


class SensorConeMap(Sensor):
    def __init__(self, env, max_objects=100, normalize=True, bbox=(-30, 30, -30, 30)):
        super(SensorConeMap, self).__init__(env)
        self._max_objects = max_objects
        self._normalize = normalize
        self._bbox = bbox
        self.imperfection_miss_chance = 0.
        self.imperfection_position_error_scale = 0.
        self.imperfection_position_error_clip = 1.
        self.imperfection_misclassify_chance = 0.

    @property
    def bbox(self):
        return tuple(self._bbox)

    @property
    def observation_space(self) -> gym.Space:
        return gym.spaces.Box(-np.inf, np.inf, shape=(self._max_objects, 1 + 4))

    @property
    def view_normalizer(self):
        return max((abs(x) for x in self._bbox))

    def observe(self, env):
        cones = env.objects['cones'].transformed(env.ego_transform).filtered_aabb(*self._bbox)

        vis_pos = cones.data[:, :2]
        cone_types_idx = np.argmax(cones.data[:, 2:], axis=-1) + 1

        # Imperfection and noise in observation model
        mask = np.random.random(len(cones)) >= self.imperfection_miss_chance
        cone_pos_offset = np.random.normal(loc=0, scale=self.imperfection_position_error_scale, size=vis_pos.shape)
        cone_pos_offset = np.clip(cone_pos_offset, -self.imperfection_position_error_clip, self.imperfection_position_error_clip)
        vis_pos = vis_pos + cone_pos_offset

        misclassified = np.random.random(len(mask)) < self.imperfection_misclassify_chance
        # # TODO: Must be updated if cone_types modified
        cone_types_idx[misclassified] = 3 - cone_types_idx[misclassified]

        vis_pos = vis_pos[mask]
        cone_types_idx = cone_types_idx[mask]

        # Normalization
        if self._normalize:
            vis_pos = vis_pos / self.view_normalizer

        count = vis_pos.shape[0]
        enc_count = min(count, self._max_objects)

        if enc_count < count:
            import warnings
            warnings.warn(f"Discarding {count - enc_count} objects because {self._max_objects = } is too low.")

        result = np.zeros((self._max_objects, cones.data.shape[1] + 1))
        result[:enc_count, 0] = 1
        result[:enc_count, 1:3] = vis_pos[:enc_count]
        result[:enc_count, 3:] = BatchedCones.categorical_from_indices(cone_types_idx[:enc_count])

        return result
