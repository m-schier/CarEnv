import numpy as np


def intersections_lines_circles(origins, ray_normals, r, centers):
    assert len(ray_normals.shape) == 2
    assert len(centers.shape) == 2
    assert len(origins) == len(ray_normals)
    assert origins.shape[-1] == ray_normals.shape[-1]
    assert centers.shape[-1] == ray_normals.shape[-1]
    assert np.asarray(r).shape == tuple()

    if len(centers) == 0:
        return np.zeros(len(origins)), np.full(len(origins), -1)

    # Find scalar projections between all lines and all circle centers
    a = ray_normals[None, :, None, :] @ centers[:, None, :, None]
    a = np.squeeze(a, (-1, -2))

    # Find the point that is closest on the ray
    close_points = origins + ray_normals * a[..., None]

    # Calculate squared distance of that point
    dist_sq = np.sum(np.square(close_points - centers[:, None]), axis=-1)

    # Only consider if close enough and in front of origin, the corner cases of intersecting back through the origin
    # is ignored
    valid = (dist_sq < r ** 2) & (a > 0.)

    # For all valid intersects calculate the actual scalar intersection, this can become less than zero if the circle
    # clips, consider this as zero
    true_dist = a[valid] - np.maximum(0., np.sqrt(r ** 2 - dist_sq[valid]))

    result = np.full_like(dist_sq, np.inf)
    result[valid] = true_dist

    best = np.argmin(result, 0)
    assert best.shape == (len(ray_normals),)

    js = np.arange(len(ray_normals))
    result_distance = result[best, js]
    result_idxs = np.where(np.isfinite(result_distance), best, -1)
    result_distance = np.where(np.isfinite(result_distance), result_distance, 0.)

    return result_distance, result_idxs


def intersections_aabb_circles(aabb, r, centers):
    assert np.asarray(aabb).shape == (4,)
    assert np.asarray(r).shape == ()
    assert len(centers.shape) == 2
    assert centers.shape[1] == 2

    x1, x2, y1, y2 = aabb

    candidate_mask = np.logical_and(
        np.logical_and(x1 - r < centers[:, 0], centers[:, 0] < x2 + r),
        np.logical_and(y1 - r < centers[:, 1], centers[:, 1] < y2 + r),
    )

    candidates = centers[candidate_mask]

    candidate_simple_match = np.logical_or(
        np.logical_and(np.logical_and(x1 < candidates[:, 0], candidates[:, 0] < x2), np.logical_and(y1 - r < candidates[:, 1], candidates[:, 1] < y2 + r)),
        np.logical_and(np.logical_and(x1 - r < candidates[:, 0], candidates[:, 0] < x2 + r), np.logical_and(y1 < candidates[:, 1], candidates[:, 1] < y2)),
    )

    # Check those without simple region match for proximity to corners to check if on rounded corners
    # TODO: Could only calculate for non-simple matches
    candidate_corner_match = np.logical_or(
        np.logical_or(
            (x1 - candidates[:, 0]) ** 2 + (y1 - candidates[:, 1]) ** 2 < r ** 2,
            (x1 - candidates[:, 0]) ** 2 + (y2 - candidates[:, 1]) ** 2 < r ** 2
        ),
        np.logical_or(
            (x2 - candidates[:, 0]) ** 2 + (y1 - candidates[:, 1]) ** 2 < r ** 2,
            (x2 - candidates[:, 0]) ** 2 + (y2 - candidates[:, 1]) ** 2 < r ** 2
        ),
    )

    result = np.zeros(len(centers), dtype=bool)
    result[candidate_mask] = np.logical_or(candidate_simple_match, candidate_corner_match)
    return result
