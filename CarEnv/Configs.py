from copy import deepcopy


VEH_CAR_KINEMATIC = {
    'type': 'bicycle',
    'wheelbase': 2.4,
}


PARALLEL_PARKING = {
    'action': {'type': 'continuous_steering_accel'},
    'longitudinal': {'type': 'simple'},
    # 'steering': 'direct()',
    'steering': 'linear(60)',
    'problem': {'type': 'parallel_parking', 'start': 'after', 'max_time': 15},
    'collision_bb': (-3.81 / 2, 3.81 / 2, -1.48 / 2, 1.48 / 2),
    'dt': .1,
    'vehicle': VEH_CAR_KINEMATIC,
    'sensors': {
        'cones_set': {
            'type': 'conemap',
            'bbox': (-30, 30, -30, 30),
        },
    }
}


VEH_CAR_DYNAMIC = {
    'type': 'dyn_dugoff',
    'wheelbase': 2.4,
    'mass': 750.,
    'inertia': 812.,  # Approximated by 0.1269*m*R*L according to "Approximation von Trägheitsmomenten bei Personenkraftwagen", Burg, 1982
    'inertia_front': 2.,  # Inertia of front axle
    'inertia_rear': 2.,   # Inertia of rear axle
    'engine_power': 60 * 1000,
    'engine_torque': 1_200.,
    'brake_torque': 3_000.,
    'brake_balance': .5,
    'mu_front': 1.05,
    'mu_rear': 1.05,
    'c_alpha_front': 10.,
    'c_sigma_front': 10.,
    'c_alpha_rear': 12.,
    'c_sigma_rear': 12.,
    'rwd': False,
}

# Racing with dynamic single track model
RACING = {
    'action': {'type': 'continuous_steering_pedals'},
    'longitudinal': {'type': 'simple'},
    'steering': 'linear(60)',
    'collision_bb': (-3.81 / 2, 3.81 / 2, -1.48 / 2, 1.48 / 2),
    'vehicle': VEH_CAR_DYNAMIC,
    'problem': {'type': 'racing', 'track_width': 8., 'cone_width': 7., 'k_forwards': .1, 'k_base': .0, 'extend': 150, 'time_limit': 60.},
    'dt': .1,
    'physics_divider': 20,
    'sensors': {
        'cones_set': {
            'type': 'conemap',
            'bbox': (-15, 45, -30, 30),
        },
    }
}


_STANDARD_ENVS = {
    'parking': PARALLEL_PARKING,
    'racing': RACING,
}


def get_standard_env_config(name):
    return deepcopy(_STANDARD_ENVS[name])


def get_standard_env_names():
    return list(_STANDARD_ENVS.keys())
