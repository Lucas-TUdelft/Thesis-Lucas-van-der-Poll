import numpy as np


def determine_burn_velocity(capsule_initial_mass):
    iterating = True

    rho_0 = 1.225  # kg/m^3
    H = 7200  # m
    h = 30000  # m
    v = -900  # m/s
    dt = 0.1
    t = 0

    descending = True
    while descending:

        rho = rho_0 * np.exp(-h / H)

        Fg = 9.81 * capsule_initial_mass
        Fd = 0.5 * 1.15 * rho * (v ** 2) * (np.pi * (2.7 ** 2))

        a = (Fd - Fg) / capsule_initial_mass
        v = v + a * dt
        h = h + v * dt

        t = t + dt

        if h <= 2000:
            descending = False

    return (v)


def determine_vehicle_mass(burn_velocity):
    final_vel = burn_velocity
    Isp = 360  # s
    mf_dry_init = 10267  # kg
    T = 980 * 1000  # N

    iterating = True

    delta_v = final_vel
    mp_old = 0.1  # kg
    mf_dry = mf_dry_init

    while iterating:
        mp = mf_dry * (np.exp(delta_v / (Isp * 9.81)) - 1)
        mi = mf_dry + mp
        a = (T / mi) - 9.81
        tb = final_vel / a
        delta_v = final_vel + tb * 9.81
        print(delta_v)

        delta_mp = mp / mp_old
        if delta_mp <= 1.01:
            iterating = False
            '''
            print('success')
            print('delta_mp: ', delta_mp)
            print('burn time: ', tb)
            print('total initial mass: ', mp + mf_dry)
            '''
        else:
            mp_old = mp
            mf_dry = mf_dry_init + mp * 0.2

    return (mp, mf_dry, mp + mf_dry)


searching = True
vehicle_mass = 19057.8
old_vehicle_mass = vehicle_mass

while searching:

    burn_velocity = -1 * determine_burn_velocity(vehicle_mass)
    propellant_mass, dry_mass, vehicle_mass = determine_vehicle_mass(burn_velocity)

    vehicle_mass_difference_ratio = (old_vehicle_mass - vehicle_mass) / old_vehicle_mass
    if abs(vehicle_mass_difference_ratio) <= 0.00001:
        print('completed, vehicle mass difference:', vehicle_mass_difference_ratio)
        print('burn_velocity:', burn_velocity, 'm/s')
        print('propellant mass:', propellant_mass, 'kg')
        print('dry mass:', dry_mass, 'kg')
        print('total initial mass:', vehicle_mass, 'kg')
        searching = False
    else:
        print('vehicle mass difference', vehicle_mass_difference_ratio)
        old_vehicle_mass = vehicle_mass