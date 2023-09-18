import numpy as np

from .BatchedObjects import BatchedObjects
from .Collision import intersections_aabb_circles


class BatchedCones(BatchedObjects):
    def __init__(self, data, hit_count=0):
        assert len(data.shape) == 2
        assert data.shape[-1] == 4

        # Features are x, y, is_blue, is_yellow
        self._data = data
        self.hit_count = hit_count

    @property
    def radius(self):
        return .3

    @property
    def centers(self):
        return self._data[:, :2]

    @staticmethod
    def get_cone_color(style):
        if style == 1:
            return 56 / 255, 103 / 255, 214 / 255
        elif style == 2:
            return 254 / 255, 211 / 255, 48 / 255
        else:
            raise ValueError(style)

    @staticmethod
    def categorical_from_indices(types):
        assert len(types.shape) == 1
        types_idx = types - 1  # Starts at 1
        assert np.all(types_idx) >= 0

        types_cat = np.zeros((len(types), 2))
        types_cat[np.arange(len(types)), types_idx] = 1.
        return types_cat

    @staticmethod
    def from_track_dict(track_dict) -> 'BatchedCones':
        pos = track_dict['cone_pos']
        types = track_dict['cone_type']
        return BatchedCones(np.concatenate([pos, BatchedCones.categorical_from_indices(types)], axis=-1))

    @property
    def data(self):
        return self._data

    def transformed(self, transform) -> 'BatchedCones':
        pos_hom = np.concatenate([self._data[:, :2], np.ones_like(self._data[:, :1])], axis=-1)
        new_pos = np.squeeze(transform @ pos_hom[..., None], -1)

        new_data = np.concatenate([new_pos[:, :2], self._data[:, 2:]], axis=-1)
        return BatchedCones(new_data, hit_count=self.hit_count)

    def filtered_aabb(self, *args) -> 'BatchedCones':
        mask = BatchedObjects.filter_mask_aabb(self._data[:, :2], *args)
        return BatchedCones(self._data[mask], hit_count=self.hit_count)

    def update(self, env):
        us_in_local = self.transformed(env.ego_transform)
        intersected = intersections_aabb_circles(env.collision_bb, self.radius, us_in_local.data[:, :2])

        hits = np.sum(intersected)
        self.hit_count += hits
        env.add_to_reward(hits * -.2)
        env.metrics['cones_hit'] = env.metrics.get('cones_hit', 0) + hits

        self._data = self._data[~intersected]
