import numpy as np

# return to landing site
delta_V = np.cos(np.deg2rad(-5.0)) * 6.93E3
M_dry = 23000
Isp = 360
g0 = 9.807
m0 = M_dry * np.exp(delta_V / (Isp * g0))
mp = m0 - M_dry
mp_total = 140000
print(f'required propellant mass: {mp} [kg]')
print(f'resultant payload mass penalty: {mp/4} [kg]')
print(f'total propellant mass fraction needed for boostback burn: {mp / mp_total}')

# downrange landing


def initial_freefall(M, burn_start):
    vx = np.cos(np.deg2rad(-5.0)) * 6.93E3
    vy = np.sin(np.deg2rad(-5.0)) * 6.93E3
    h = 157 * 10 ** 3
    g = 9.81
    t = 0.0
    dt = 0.1

    while h >= burn_start * 10 ** 3:

        ay = -g
        ax = 0

        vy += ay * dt
        vx += ax * dt

        h += vy * dt

        t += dt

    return(vy, vx, t)

def entry_burn(vy, vx, t, M, mp, thrust_level, burn_start, burn_angle):
    print(t)
    h = burn_start * 10 ** 3
    Isp = 360
    g0 = 9.807
    ran_out_of_fuel = False
    max_q_dot_exceeded = False
    checking_qdot = True
    dt = 0.1

    rho_0 = 1.225  # kg/m^3
    H = 7200  # m

    descending = True
    while descending:
        rho = rho_0 * np.exp(-h / H)

        m_dot = 980 * 1000 * thrust_level / (Isp * g0)
        #print(m_dot)

        flight_path_angle = np.arctan2(vy,vx)

        Fg = g0 * M
        Fdy = 0.5 * 0.82 * rho * (vy ** 2) * (np.pi * (2.7 ** 2))
        Fdx = 0.5 * 0.82 * rho * (vx ** 2) * (np.pi * (2.7 ** 2))
        #print('Fdx', Fdx)
        if mp > 0.0:
            FTy = np.sin(burn_angle) * (980 * 1000 * thrust_level)
            #FTy = np.sin(-flight_path_angle) * (980 * 1000 * thrust_level)
            #print('FTy', FTy)
            FTx = - np.cos(-burn_angle) * (980 * 1000 * thrust_level)
            #FTx = - np.cos(-flight_path_angle) * (980 * 1000 * thrust_level)
            #print('FTx', FTx)
            mp +=  - m_dot * dt
            #print(mp)
            M += - m_dot * dt
        elif thrust_level > 0.0:
            if ran_out_of_fuel == False:
                print('ran out of fuel')
            ran_out_of_fuel = True
            #print('ran out of fuel')
            FTy = 0.0
            FTx = 0.0

        #print(np.sqrt((FTy ** 2) + (FTx ** 2)))
        #print(Fg)
        ay = (Fdy + FTy - Fg) / M
        ax = (- Fdx + FTx) / M
        #print(Fdy, FTy, Fg, ay)
        #print('accelerations', ay, ax)

        vy += ay * dt
        vx += ax * dt
        #print('velocities', vy, vx)
        #print(np.sqrt((vx ** 2) + (vy ** 2)))

        h += vy * dt
        #print(h)

        t += dt

        q_dot = (1 / np.sqrt(0.7)) * (1.83 * 10 ** (-4)) * np.sqrt(rho) * (np.sqrt((vx**2) + (vy**2)) ** 3)
        #print(q_dot)
        if q_dot >= 200000 and checking_qdot == True:
            print(q_dot)
            print(t)

            #print('exceeded q_dot max')
            if mp <= 0.0:
                a = 1
                #print('ran out of fuel')
            max_q_dot_exceeded = True
            checking_qdot = False
            descending = False

        if h <= 40000:
            descending = False

    return(vy, vx, t, ran_out_of_fuel, max_q_dot_exceeded)


def determine_burn_velocity(vx, vy, M, t):
    iterating = True

    rho_0 = 1.225  # kg/m^3
    H = 7200  # m
    h = 40000  # m
    dt = 0.1

    descending = True
    while descending:

        rho = rho_0 * np.exp(-h / H)

        Fg = 9.81 * M
        Fdy = 0.5 * 0.82 * rho * (vy ** 2) * (np.pi * (2.7 ** 2))
        Fdx = 0.5 * 0.82 * rho * (vx ** 2) * (np.pi * (2.7 ** 2))

        ay = (Fdy - Fg) / M
        ax = -Fdx / M

        vy += ay * dt
        vx += ax * dt

        h += vy * dt

        t = t + dt

        if h <= 2000:
            descending = False

    v = np.sqrt((vx ** 2) + (vy ** 2))
    return(v)

M_dry = 23000
M_p = 21650 / 4
thrust_level = 1.0
iterating = True
M = M_dry + M_p
burn_starts = [150, 140, 130, 120, 110, 100, 90, 80]
burn_angles = [5, 11, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
for i in range(len(burn_starts)):
    print(f'burn start at altitude {burn_starts[i]} km')
    for j in range(len(burn_angles)):
        print(f'burn angle {burn_angles[j]} deg')
        vy, vx, t = initial_freefall(M, burn_starts[i])
        vy, vx, t, ran_out_of_fuel, max_q_dot_exceeded = entry_burn(vy, vx, t, M, M_p, thrust_level, burn_starts[i], np.deg2rad(burn_angles[j]))
        if not max_q_dot_exceeded:
            print('SUCCESSFUL REENTRY BURN')
'''
while iterating:
    M = M_dry + M_p
    vy, vx, t = initial_freefall(M)
    vy, vx, t, ran_out_of_fuel, max_q_dot_exceeded = entry_burn(vy, vx, t, M, M_p, thrust_level)
    if max_q_dot_exceeded and ran_out_of_fuel:
        M_p += 10
        print('new M_p:', M_p)
    if max_q_dot_exceeded and not ran_out_of_fuel:
        thrust_level += 0.01
        print('new thrust_level:', thrust_level)
    if not max_q_dot_exceeded:
        v_final = determine_burn_velocity(vx, vy, M, t)
'''


