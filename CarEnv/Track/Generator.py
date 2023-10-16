import numpy as np
from shapely.geometry import MultiPoint, LinearRing, LineString, Point, Polygon
import matplotlib.pyplot as plt


class BadTrackException(Exception):
    pass


def plot_line_string_ring(points, *args, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    points = np.array(points)
    points = np.concatenate([points, points[0:1]], axis=0)
    ax.plot(*points.T, *args, **kwargs)


# Plots a Polygon to pyplot `ax`
def plot_polygon(poly, ax=None, **kwargs):
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    from matplotlib.collections import PatchCollection

    if ax is None:
        ax = plt.gca()

    path = Path.make_compound_path(
        Path(np.asarray(poly.exterior.coords)[:, :2]),
        *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors])

    patch = PathPatch(path, **kwargs)
    collection = PatchCollection([patch], **kwargs)

    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection


def push_apart(pts, dist=.05, reps=3):
    for _ in range(reps):
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                h = pts[j] - pts[i]
                h_dist = np.linalg.norm(h)
                if h_dist < dist:
                    h = h / h_dist * (dist - h_dist)
                    pts[j] += h
                    pts[i] -= h


def make_non_convex(pts, offset=.2, dist=.05, rng=None):
    # If given a closed path, remove the end-point, we already handle it, and it will cause issues
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]

    if rng is None:
        rng = np.random.default_rng()

    # positive rolls towards right, i.e. roll([1, 2, 3], 1) -> [3, 1, 2]
    # negative rolls towards left, i.e. roll([1, 2, 3], -1) -> [2, 3, 1]
    lerps = rng.random((len(pts), 1)) * .5 + .25
    lerp_points = (pts * lerps + np.roll(pts, -1, axis=0) * (1 - lerps))
    dists = pts - np.roll(pts, -1, axis=0)
    orthos = np.stack([dists[:, 1], -dists[:, 0]], axis=-1)
    norms = np.linalg.norm(orthos, axis=-1)
    orthos /= norms[:, None]
    new_pts = lerp_points + orthos * (rng.random(size=(len(pts), 1)) * 2 * offset - offset)

    result = np.empty((len(pts) * 2, 2), dtype=pts.dtype)
    result[::2] = pts
    result[1::2] = new_pts

    push_apart(result, dist=dist)

    return result


def catmull_rom_closed(points, alpha=.5, num_points=20):
    if np.allclose(points[0], points[-1]):
        raise ValueError("Unexpected repetition of start point")

    p0 = np.roll(points, 2, axis=0)
    p1 = np.roll(points, 1, axis=0)
    p2 = points
    p3 = np.roll(points, -1, axis=0)

    def tj(ti, pi, pj):
        delta = pi - pj
        lengths = np.linalg.norm(delta, axis=-1)
        return ti + lengths ** alpha

    t0 = np.zeros(len(points))
    t1 = tj(t0, p0, p1)
    t2 = tj(t1, p1, p2)
    t3 = tj(t2, p2, p3)

    t0 = t0[:, None, None]
    t1 = t1[:, None, None]
    t2 = t2[:, None, None]
    t3 = t3[:, None, None]

    t = np.linspace(0, 1, num_points)[:, None] * (t2 - t1) + t1
    # t shape is (num_segments, num_points, 1) with last reserved for num_dimensions=2

    a1 = (t1 - t) / (t1 - t0) * p0[:, None, :] + (t - t0) / (t1 - t0) * p1[:, None, :]
    a2 = (t2 - t) / (t2 - t1) * p1[:, None, :] + (t - t1) / (t2 - t1) * p2[:, None, :]
    a3 = (t3 - t) / (t3 - t2) * p2[:, None, :] + (t - t2) / (t3 - t2) * p3[:, None, :]
    b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
    b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
    points = (t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2
    return points.reshape(-1, 2)


def make_track_centerline(extends, min_dist=15., offset=20., rng=None):
    if rng is None:
        rng = np.random.default_rng()

    points = rng.uniform(0, 1, size=(rng.integers(10, 20, endpoint=True), 2)) * extends
    push_apart(points, dist=min_dist)
    hull = np.array(MultiPoint(points).convex_hull.exterior.coords.xy).T
    track_pts = make_non_convex(hull, offset=offset, dist=min_dist, rng=rng)
    track_cm = catmull_rom_closed(track_pts)
    return track_cm


def check_valid_track(centerline, width):
    lr = LinearRing(centerline)

    if not lr.is_simple:
        from shapely.validation import explain_validity
        raise BadTrackException(f"Not simple: {explain_validity(lr)}")

    polygon = lr.buffer(width / 2)

    if type(polygon) != Polygon:
        # Can also be MultiPolygon, which is certainly false
        raise BadTrackException("Centerline is MultiPolygon")

    if len(polygon.interiors) != 1:
        raise BadTrackException("Centerline has more than one interiors")

    return True


def make_cones_and_start_pose(centerline, width):
    from .Util import discretize_contour

    poly = make_polygon_from_centerline(centerline, width)

    if type(poly) is not Polygon:
        raise BadTrackException("Extended centerline is not Polygon")

    outer = discretize_contour(np.asarray(poly.exterior.coords)[:, :2], 5.)
    inner = discretize_contour(np.asarray(poly.interiors[0].coords)[:, :2], 5.)

    cone_pos = np.concatenate([outer, inner])
    cone_type = np.concatenate([np.full(len(outer), 1, dtype=int), np.full(len(inner), 2, dtype=int)])

    # Find start pos
    forward = centerline[1] - centerline[0]
    forward = forward / np.linalg.norm(forward)

    theta = np.arctan2(-forward[1], -forward[0])

    xy = centerline[0]

    return cone_pos, cone_type, xy, theta


def make_full_environment(extends=(200, 200), width=5., cone_width=5., rng=None):
    centerline = None

    if rng is None:
        rng = np.random.default_rng()

    while True:
        try:
            centerline = make_track_centerline(extends, rng=rng)
            check_valid_track(centerline, width)
            cone_pos, cone_type, xy, theta = make_cones_and_start_pose(centerline, cone_width)
            break
        except BadTrackException:
            pass

    # Maybe reverse
    if rng.integers(0, 2) == 0:
        centerline = centerline[::-1]
        theta += np.pi
        # Map 1 -> 2 and 2 -> 1
        cone_type = 3 - cone_type

    return {
        'centerline': centerline,
        'length': LinearRing(centerline).length,
        'width': width,
        'start_xy': xy,
        'start_theta': theta,
        'cone_pos': cone_pos,
        'cone_type': cone_type
    }


def make_polygon_from_centerline(centerline, width=5.):
    poly = LinearRing(centerline).buffer(width / 2)

    return poly


def write_tum_track(track, path, total_width=7.):
    with open(path, 'w') as fp:
        print("# x_m,y_m,w_tr_right_m,w_tr_left_m", file=fp)

        for x, y in track:
            print("{},{},{},{}".format(x, y, total_width / 2, total_width / 2), file=fp)


def main():
    # Based on http://blog.meltinglogic.com/2013/12/how-to-generate-procedural-racetracks/

    points = np.random.uniform(0, 1, size=(np.random.randint(10, 20), 2)) * (200, 200)
    plt.plot(*points.T, 'bo')
    push_apart(points)
    plt.plot(*points.T, 'ro')
    hull = np.array(MultiPoint(points).convex_hull.exterior.coords.xy).T
    plot_line_string_ring(hull)
    track_pts = make_non_convex(hull, offset=30, dist=20)
    track_cm = catmull_rom_closed(track_pts)
    lr = LinearRing(track_cm).buffer(3.5)
    write_tum_track(track_cm, "..\\..\\gen.csv")
    plot_line_string_ring(track_pts, 'k-')
    plot_polygon(lr, facecolor='lightgray', edgecolor='red')

    fig = plt.figure()
    plot_polygon(lr, facecolor='black', edgecolor='black')
    plt.axis('equal')
    plt.savefig('track.png')

    plt.axis('equal')

    fig, axs = plt.subplots(4, 4)

    for ax in axs.flatten():
        t = make_track_centerline((200, 200))
        plot_line_string_ring(t, 'k-', ax=ax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

    plt.show()


if __name__ == '__main__':
    main()
