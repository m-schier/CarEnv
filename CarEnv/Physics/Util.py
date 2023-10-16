import numpy as np


def rot_mat(angle):
    assert np.shape(angle) == ()
    return rot_mats(np.array([angle]))[0]


def rot_mats(angles):
    assert len(angles.shape) == 1.

    cs = np.cos(angles)
    sn = np.sin(angles)

    # Faster than double np.stack
    result = np.empty((angles.shape[0], 2, 2))
    result[:, 0, 0] = cs
    result[:, 1, 1] = cs
    result[:, 0, 1] = sn
    result[:, 1, 0] = -sn

    assert result.shape == (angles.shape[0], 2, 2)
    return result


def soft_sign(xs, slope=10.):
    xs = xs * slope
    # es = np.exp(xs)
    # frac = (es / (es + 1.))
    frac = 1. / (1. + np.exp(-xs))
    return frac * 2. - 1.
