import numpy as np


class BatchedObjects:
    @property
    def features_dim(self):
        count, n_features = self.data.shape
        return n_features

    @property
    def data(self):
        raise NotImplementedError

    def transformed(self, transform) -> 'BatchedObjects':
        raise NotImplementedError

    @staticmethod
    def filter_mask_aabb(pos, x_min, x_max, y_min, y_max):
        return np.logical_and(
            np.logical_and(pos[:, 0] > x_min, pos[:, 0] < x_max),
            np.logical_and(pos[:, 1] > y_min, pos[:, 1] < y_max)
        )

    def filtered_aabb(self, x_min, x_max, y_min, y_max) -> 'BatchedObjects':
        raise NotImplementedError

    def __len__(self):
        count, n_features = self.data.shape
        return count

    def update(self, env):
        pass
