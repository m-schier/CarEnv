import numpy as np
from numba import jit


@jit('f8[:, :](f8[:, :], f8, b1[:])', nopython=True)
def clip_norm(vs, max_norm, is_clipped):
    assert len(vs.shape) == 2
    assert vs.shape[-1] == 2
    result = vs.copy()

    # Much faster than np.sum(np.square(vs), -1)...
    sq_norms = vs[:, 0] * vs[:, 0] + vs[:, 1] * vs[:, 1]
    is_clipped[:] = sq_norms > max_norm * max_norm
    coeffs = np.sqrt(sq_norms[is_clipped]) / max_norm

    result[is_clipped, 0] /= coeffs
    result[is_clipped, 1] /= coeffs
    return result


@jit('f8[:, :](f8[:, :], f8, f8, b1[:])', nopython=True)
def clip_opt_sat(vs, sat, opt, is_clipped):
    result = vs.copy()
    norms = np.sqrt(vs[:, 0] * vs[:, 0] + vs[:, 1] * vs[:, 1])

    # Calculate opt curve
    a = .5 / (sat - opt)
    b = -2. * a * opt
    c = sat - a * (sat ** 2) - b * sat

    sat_mask = norms > sat
    norms_sat = norms[sat_mask]

    eff_mul = np.maximum(a * norms_sat ** 2 + b * norms_sat + c, .5 * sat)
    is_clipped[:] = norms > opt  # Set the clipping info based on exceeding optimality

    result[sat_mask, 0] = result[sat_mask, 0] / norms_sat * eff_mul
    result[sat_mask, 1] = result[sat_mask, 1] / norms_sat * eff_mul

    return result


@jit('f8[:](f8[:], f8, f8, f8)', nopython=True)
def tire_curve(slip_angle, downforce, max_grip, stiffness):
    sat = max_grip / (-stiffness)

    # Clip tire curve here. This clips twice if using _tire_force, once on component and L2, but plays
    # nicely with dampening
    clipped_angle = np.minimum(sat, np.maximum(-sat, slip_angle))
    force = stiffness * clipped_angle * downforce

    return force


@jit('f8[:, :](b1[:], f8[:, :], f8[:], f8, f8[:], f8, f8, f8, f8)', nopython=True)
def fast_tire_forces(is_slip, v_loc, long_force, downforce, slip_angle, stiffness, max_grip_lin, max_grip_opt, C_r):
    # Roll resistance constant, but dampen around 0 for stability without a more intelligent integration method
    vx_es = np.exp(v_loc[:, 0] * 10.)
    vx_soft = (vx_es / (vx_es + 1.)) * 2. - 1.
    vy_es = np.exp(v_loc[:, 1] * 5.)
    vy_soft = (vy_es / (vy_es + 1.)) * 2. - 1.

    roll_resists = -C_r * vx_soft * downforce
    # Also dampen lateral forces around 0 for stability of simulation
    lat_dampen = np.abs(vy_soft)

    # Use tire curve for lateral force
    max_grip_for_tc = max_grip_lin if max_grip_opt <= 0. else max_grip_lin * 3.
    lat_force = tire_curve(slip_angle, downforce, max_grip_for_tc, stiffness) * lat_dampen

    # Faster this way than stacking...
    result = np.empty((slip_angle.shape[0], 2))
    result[:, 0] = long_force + roll_resists
    result[:, 1] = lat_force

    # Kammerscher Kreis, kind of
    if max_grip_opt <= 0.:
        result = clip_norm(result, downforce * max_grip_lin, is_slip)
    else:
        result = clip_opt_sat(result, max_grip_lin * downforce, max_grip_opt * downforce, is_slip)

    return result


def tire_forces(v_loc, long_force, downforce, slip_angle, stiffness, max_grip, C_r, max_grip_opt=0.):
    is_slip = np.empty(len(v_loc), dtype=bool)
    result = fast_tire_forces(is_slip, v_loc, long_force, downforce, slip_angle, stiffness, max_grip, max_grip_opt, C_r)
    return result, is_slip
