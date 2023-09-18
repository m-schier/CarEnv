import numpy as np


def make_orthogonals(track):
    raise NotImplementedError("Deprecated")


def track_from_tum_file(path):
    data = np.loadtxt(path, comments='#', delimiter=',')
    track = data[:, :2]
    bounds = np.stack([-data[:, 3], data[:, 2]], axis=-1)
    return track, bounds


def discretize_contour(track, step=10.):
    from shapely.geometry import LineString, Point, MultiPoint

    lines = [LineString([track[i], track[i+1]]) for i in range(len(track) - 1)]

    # Add final loop-closing line
    lines.append(LineString([track[-1], track[0]]))

    current_xy = track[0]
    result = [current_xy]
    curr_i = 0

    while True:
        circ = Point(current_xy).buffer(step).exterior

        for i in range(curr_i, len(lines)):
            intersection = lines[i].intersection(circ)
            if intersection:
                if isinstance(intersection, Point):
                    ps = np.array(intersection.xy).T
                elif isinstance(intersection, MultiPoint):
                    ps = np.concatenate([np.array(p.xy).T for p in intersection.geoms])
                else:
                    raise TypeError(str(type(intersection)))

                ld = np.array(lines[i].xy).T

                if i == curr_i:
                    max_advance = np.linalg.norm(current_xy - ld[0])
                else:
                    max_advance = -1
                best_p = None

                for p in ps:
                    assert p.shape == (2,)
                    advance = np.linalg.norm(p - ld[0])
                    if advance <= max_advance:
                        continue
                    max_advance = advance
                    best_p = p

                if best_p is not None:
                    current_xy = best_p
                    result.append(best_p)
                    curr_i = i
                    break
        else:
            # TODO: Handle end point correctly, i.e. check distance to previous
            # result.append(track[-1])
            return np.array(result)


def track_from_image(path, scale=.3, step=4.):
    import cv2

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 1:
        c = contours[0]
    elif len(contours) == 2:
        # Closed loop track
        parent_idx = np.argmin(np.array(hierarchy)[..., -1])
        c = contours[parent_idx]
    else:
        raise ValueError("Bad contour number: {}".format(len(contours)))

    c = np.squeeze(c, 1).astype(np.double) * scale
    t = discretize_contour(c, step)

    # Viz
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # plt.plot(*c.T, 'o', color='k')
    # plt.plot(*t.T, 'o', color='tab:red')
    # plt.plot(*c[0], 'x')
    # plt.plot(*c[-1], 'x')
    # plt.axis('equal')
    # plt.show()

    assert is_track_closed(t)
    return t


def is_track_closed(track):
    # TODO: Should also check angle
    max_dist = np.max(np.linalg.norm(track[1:] - track[:-1], axis=-1))
    close_dist = np.linalg.norm(track[0] - track[-1])
    return close_dist < max_dist * 2


def make_true_normals(track, is_closed=False):
    t_next = np.roll(track, 1, axis=0)
    t_prev = np.roll(track, -1, axis=0)

    normals = t_next - t_prev
    normals = np.stack([-normals[:, 1], normals[:, 0]], axis=-1)
    normals /= np.linalg.norm(normals, axis=-1)[:, None]

    if not is_closed:
        normals[0] = normals[1]
        normals[-1] = normals[-2]

    return normals


def annotate_scale(h, ax=None, text=None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    if ax is None:
        ax = plt.gca()

    if text is None:
        text = "h = {:.3}".format(h)

    artist_scale = AnchoredSizeBar(ax.transData, h, text,
                                   'upper right', label_top=True, borderpad=1, frameon=False)
    ax.add_artist(artist_scale)
    return artist_scale
