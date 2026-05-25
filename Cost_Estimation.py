import matplotlib.pyplot as plt
import numpy as np

def reusable_vehicle_cost_variable(launch_number, t_travel, M_p, n_reuses):

    #for i in range(launch_number):
    #    print(i + 1)
    n_p = ((launch_number - 1) // n_reuses) + 1
    TMY_rH = 2080
    LpA = 30
    CMY = 286.425

    M_expendable = 9664 - 2789  # kg
    M_resuable = 10330.45       # kg
    C_stage = 3.7 * 10**6       # ME
    inflation_2017_2025 = 1.3022
    inflation_2015_2025 = 1.3277
    r = 6

    f_4 = 0.85**(np.log10(n_p) / np.log10(2))


    # reusable vehicle cost (2017)
    F_sr_2017 = 2 * (M_resuable / M_expendable) * f_4 * C_stage
    #print(F_sr_2017)
    F_sr = F_sr_2017 * inflation_2017_2025
    #print(f'first production cost: {F_sr/1000000} M€')


    # rest of the core stage cost (recent enough)
    M_stage1 = 23000 # kg
    M_remainder = M_stage1 - M_expendable
    C_remainder = 26.78205719 * 1000000

    # transport (recent enough)

    C_ship = (1.2 * t_travel * 20 / TMY_rH) + (760000 / (LpA * CMY))

    C_ship_euro = C_ship * CMY
    #print(f'Transportation Costs: {C_ship_euro/1000000} M€')

    # refurbishment (recent enough)

    n_refurbishment_current_cycle = (launch_number - n_p - ((n_p - 1) * (n_reuses - 1))) + 1
    print(launch_number, n_p, ((n_p - 1) * (n_reuses - 1)))
    print(n_refurbishment_current_cycle)
    #print(n_refurbishment_current_cycle)

    C_refurbishment_previous_cycles_single = 0
    for i in range(n_reuses - 1):
        C_refurbishment = F_sr * (0.25 * (i + 1) **(np.log(1.15)/np.log(2)))
        C_refurbishment_previous_cycles_single += C_refurbishment
    C_refurbishment_previous_cycles = C_refurbishment_previous_cycles_single * (n_p - 1)
    '''
    C_refurbishment_previous_cycles = (F_sr * (0.25 * 1 **(np.log(1.15)/np.log(2))) + \
                                      F_sr * (0.25 * 2 **(np.log(1.15)/np.log(2))) + \
                                      F_sr * (0.25 * 3 **(np.log(1.15)/np.log(2))) + \
                                      F_sr * (0.25 * 4 **(np.log(1.15)/np.log(2)))) * (n_p - 1)
    '''
    C_refurbishment_current_cycle = 0
    for i in range(n_refurbishment_current_cycle):
        C_refurbishment_current_cycle += F_sr * (0.25 * i **(np.log(1.15)/np.log(2)))
    '''
    if n_refurbishment_current_cycle == 1:
        #print('newly built stage')
        C_refurbishment_current_cycle = 0
    elif n_refurbishment_current_cycle == 2:
        #print('+1 refurbishments')
        C_refurbishment_current_cycle = F_sr * (0.25 * 1 **(np.log(1.15)/np.log(2)))
    elif n_refurbishment_current_cycle == 3:
        #print('+2 refurbishments')
        C_refurbishment_current_cycle = F_sr * (0.25 * 1 **(np.log(1.15)/np.log(2))) + \
                                        F_sr * (0.25 * 2 **(np.log(1.15)/np.log(2)))
    elif n_refurbishment_current_cycle == 4:
        #print('+3 refurbishments')
        C_refurbishment_current_cycle = F_sr * (0.25 * 1 **(np.log(1.15)/np.log(2))) + \
                                        F_sr * (0.25 * 2 **(np.log(1.15)/np.log(2))) + \
                                        F_sr * (0.25 * 3 **(np.log(1.15)/np.log(2)))
    else:
        #print('+4 refurbishments')
        C_refurbishment_current_cycle = F_sr * (0.25 * 1 ** (np.log(1.15) / np.log(2))) + \
                                        F_sr * (0.25 * 2 ** (np.log(1.15) / np.log(2))) + \
                                        F_sr * (0.25 * 3 ** (np.log(1.15) / np.log(2))) + \
                                        F_sr * (0.25 * 4 ** (np.log(1.15) / np.log(2)))
    C_refurbishment = C_refurbishment_previous_cycles + C_refurbishment_current_cycle
    #print(f'Refurbishment Costs: {C_refurbishment/1000000} M€')
    '''
    C_refurbishment = C_refurbishment_previous_cycles + C_refurbishment_current_cycle

    # propellant cost (2015)

    C_propellant_2015 = (M_p / (r + 1)) * 1.35 + (M_p - (M_p / (r + 1))) * 0.14
    C_propellant = inflation_2015_2025 * C_propellant_2015
    #print(f'Propellant Costs: {C_propellant/1000000} M€')

    #print(F_sr)
    print(f'launch number: {launch_number}, stage productions: {n_p}, stage refurbishments: {launch_number - n_p}')
    total_cost = (n_p * F_sr + (C_propellant + C_ship_euro) * (launch_number - n_p) + C_refurbishment) / 1000000
    return(total_cost)

def reusable_vehicle_cost(launch_number, t_travel, M_p):

    #for i in range(launch_number):
    #    print(i + 1)
    n_p = ((launch_number - 1) // 5) + 1
    TMY_rH = 2080
    LpA = 30
    CMY = 286.425

    M_expendable = 9664 - 2789  # kg
    M_resuable = 10330.45       # kg
    C_stage = 3.7 * 10**6       # ME
    inflation_2017_2025 = 1.3022
    inflation_2015_2025 = 1.3277
    r = 6

    f_4 = 0.85**(np.log10(n_p) / np.log10(2))


    # reusable vehicle cost (2017)
    F_sr_2017 = 2 * (M_resuable / M_expendable) * f_4 * C_stage
    #print(F_sr_2017)
    F_sr = F_sr_2017 * inflation_2017_2025
    #print(f'first production cost: {F_sr/1000000} M€')


    # rest of the core stage cost (recent enough)
    M_stage1 = 23000 # kg
    M_remainder = M_stage1 - M_expendable
    C_remainder = 26.78205719 * 1000000

    # transport (recent enough)

    C_ship = (1.2 * t_travel * 20 / TMY_rH) + (760000 / (LpA * CMY))

    C_ship_euro = C_ship * CMY
    #print(f'Transportation Costs: {C_ship_euro/1000000} M€')

    # refurbishment (recent enough)

    n_refurbishment_current_cycle = (launch_number - n_p - ((n_p - 1) * 4)) + 1
    #print(n_refurbishment_current_cycle)

    C_refurbishment_previous_cycles = (F_sr * (0.25 * 1 **(np.log(1.15)/np.log(2))) + \
                                      F_sr * (0.25 * 2 **(np.log(1.15)/np.log(2))) + \
                                      F_sr * (0.25 * 3 **(np.log(1.15)/np.log(2))) + \
                                      F_sr * (0.25 * 4 **(np.log(1.15)/np.log(2)))) * (n_p - 1)
    if n_refurbishment_current_cycle == 1:
        #print('newly built stage')
        C_refurbishment_current_cycle = 0
    elif n_refurbishment_current_cycle == 2:
        #print('+1 refurbishments')
        C_refurbishment_current_cycle = F_sr * (0.25 * 1 **(np.log(1.15)/np.log(2)))
    elif n_refurbishment_current_cycle == 3:
        #print('+2 refurbishments')
        C_refurbishment_current_cycle = F_sr * (0.25 * 1 **(np.log(1.15)/np.log(2))) + \
                                        F_sr * (0.25 * 2 **(np.log(1.15)/np.log(2)))
    elif n_refurbishment_current_cycle == 4:
        #print('+3 refurbishments')
        C_refurbishment_current_cycle = F_sr * (0.25 * 1 **(np.log(1.15)/np.log(2))) + \
                                        F_sr * (0.25 * 2 **(np.log(1.15)/np.log(2))) + \
                                        F_sr * (0.25 * 3 **(np.log(1.15)/np.log(2)))
    else:
        #print('+4 refurbishments')
        C_refurbishment_current_cycle = F_sr * (0.25 * 1 ** (np.log(1.15) / np.log(2))) + \
                                        F_sr * (0.25 * 2 ** (np.log(1.15) / np.log(2))) + \
                                        F_sr * (0.25 * 3 ** (np.log(1.15) / np.log(2))) + \
                                        F_sr * (0.25 * 4 ** (np.log(1.15) / np.log(2)))
    C_refurbishment = C_refurbishment_previous_cycles + C_refurbishment_current_cycle
    #print(f'Refurbishment Costs: {C_refurbishment/1000000} M€')


    # propellant cost (2015)

    C_propellant_2015 = (M_p / (r + 1)) * 1.35 + (M_p - (M_p / (r + 1))) * 0.14
    C_propellant = inflation_2015_2025 * C_propellant_2015
    #print(f'Propellant Costs: {C_propellant/1000000} M€')

    #print(f'launch number: {launch_number}, stage productions: {n_p}, stage refurbishments: {launch_number - n_p}')
    total_cost = (n_p * F_sr + (C_propellant + C_ship_euro) * (launch_number - n_p) + C_refurbishment) / 1000000
    return(total_cost)

target_location = 'Cabo Verde'
if target_location == 'Natal':
    t_travel = (11 * 24) + 23 + (3 / 60) # h
    M_p = (18558.58 + 317) * 1.1 # kg
    payload_mass_penalty = (10648.25 + (18558.58 * 1.2) - 6875) * 0.25
    #print(payload_mass_penalty)
if target_location == 'Cabo Verde':
    t_travel = (7 * 24) + 17 + (11 / 60) # h
    M_p = (15703.60 + 317) * 1.1 #kg
    payload_mass_penalty = (10648.25 + (15703.60 * 1.2) - 6875) * 0.25
    #print(payload_mass_penalty)
if target_location == 'Canarias':
    t_travel = (4 * 24) + 20 + (24 / 60) # h
    M_p = (38084.08 + 317) * 1.1 #kg
    payload_mass_penalty = (10648.25 + (38084.08 * 1.2) - 6875) * 0.25
    #print(payload_mass_penalty)


T1 = 14479579.45
MPA = T1 * 0.1
FM1 = T1 - MPA
STH = 3.1
L_d = 1.0
HW = 1.0
C_MAIT = FM1 * STH * L_d * HW
C_ENG = 3.0 * FM1
development_cost = (C_ENG + (C_ENG + C_MAIT) * 0.1 + C_MAIT) / 1000000

account_for_5th_launch_expendable = False
checking_A62_per_launch = True
checking_A64_per_launch = True

A62_reusable_list = []
A64_reusable_list = []
launches_list = []

n_reuses_list = [4,5,6,7,8,9,10]
for j in range(len(n_reuses_list)):
    n_reuses = n_reuses_list[j]
    A62_expendable = []
    A64_expendable = []
    A62_reusable = []
    A64_reusable = []
    launches = []
    for i in range(500):
        total_cost = reusable_vehicle_cost_variable(i+1, t_travel, M_p, n_reuses)
        if (i+1) <= 150:
            development_cost_per_launch = development_cost / 150
            total_cost += development_cost_per_launch * (i+1)
        #total_cost_2 = reusable_vehicle_cost(i+1, t_travel, M_p)
        #print(total_cost - total_cost_2)
        #print(total_cost)
        non_reusable_cost = (3.7 + 1.0) * i

        Ariane62_expendable = 100 - 10 + 1                          # Cost - Vulcain + Prometheus
        Ariane62_minus_engine_bay = Ariane62_expendable - (3.7 + 1)
        Ariane64_expendable = 115 - 10 + 1
        Ariane64_minus_engine_bay = Ariane64_expendable - (3.7 + 1)

        cost_per_launch_A62 = (total_cost / (i+1)) + Ariane62_minus_engine_bay
        cost_per_launch_A64 = (total_cost / (i+1)) + Ariane64_minus_engine_bay

        if checking_A62_per_launch:
            if Ariane62_expendable >= cost_per_launch_A62:
                print(f'launch at which cost per launch of A62 becomes less in reusable case: {i + 1}')
                checking_A62_per_launch = False
        if checking_A64_per_launch:
            if Ariane64_expendable >= cost_per_launch_A64:
                print(f'launch at which cost per launch of A64 becomes less in reusable case: {i + 1}')
                checking_A64_per_launch = False

        A62_expendable.append(Ariane62_expendable)
        A64_expendable.append(Ariane64_expendable)
        A62_reusable.append(cost_per_launch_A62)
        A64_reusable.append(cost_per_launch_A64)
        launches.append(i+1)

        print('########')

    A62_reusable_list.append(A62_reusable)
    A64_reusable_list.append(A64_reusable)
    launches_list.append(launches)

        #n = 0
        #launch_number = i + 1
        #n_p = ((launch_number - 1) // 5) + 1
        #n_refurbishment_current_cycle = (launch_number - n_p - ((n_p - 1) * 4)) + 1

    '''
    print(f'number of full refurbishments from previous cycles: {n_p - 1}')
    n = 4 * (n_p - 1)
    if n_refurbishment_current_cycle == 1:
        print('newly built stage')
        n = n
    elif n_refurbishment_current_cycle == 2:
        print('+1 refurbishments')
        n = n + 1
    elif n_refurbishment_current_cycle == 3:
        print('+2 refurbishments')
        n = n + 2
    elif n_refurbishment_current_cycle == 4:
        print('+3 refurbishments')
        n = n + 3
    else:
        print('+4 refurbishments')
        n = n + 4

    print(f'number of refurbishments: {n}')
    '''


plt.suptitle(target_location)
for i in range(len(n_reuses_list)):
    plt.plot(launches, A62_reusable_list[i], label=f'Ariane62 reusable, {n_reuses_list[i]} reuses')
    plt.plot(launches, A64_reusable_list[i], label=f'Ariane64 reusable, {n_reuses_list[i]} reuses')
plt.plot(launches, A62_expendable, label='Ariane62 expendable')
plt.plot(launches, A64_expendable, label='Ariane64 expendable')
plt.xlabel('number of launches [-]')
plt.ylabel('Cost per Launch [M€]')
plt.legend()
plt.grid()
plt.show()

plt.suptitle(target_location)
for j in range(len(n_reuses_list)):
    A62_expendable_cumulative = []
    A64_expendable_cumulative = []
    A62_reusable_cumulative = []
    A64_reusable_cumulative = []
    A62_expendable_cumulative_i = 0
    A64_expendable_cumulative_i = 0
    A62_reusable_cumulative_i = 0
    A64_reusable_cumulative_i = 0
    checking_A62 = True
    checking_A64 = True
    for i in range(len(A62_expendable)):
        A62_expendable_cumulative_i += A62_expendable[i]
        A64_expendable_cumulative_i += A64_expendable[i]
        A62_reusable_cumulative_i += A62_reusable_list[j][i]
        A64_reusable_cumulative_i += A64_reusable_list[j][i]
        if checking_A62:
            if A62_expendable_cumulative_i >= A62_reusable_cumulative_i:
                print(f'launch at which cumulative cost of A62 becomes less in reusable case: {i + 1}')
                checking_A62 = False
        if checking_A64:
            if A64_expendable_cumulative_i >= A64_reusable_cumulative_i:
                print(f'launch at which cumulative cost of A64 becomes less in reusable case: {i + 1}')
                checking_A64 = False

        A62_expendable_cumulative.append(A62_expendable_cumulative_i)
        A64_expendable_cumulative.append(A64_expendable_cumulative_i)
        A62_reusable_cumulative.append(A62_reusable_cumulative_i)
        A64_reusable_cumulative.append(A64_reusable_cumulative_i)

    plt.plot(launches, A62_reusable_cumulative, label=f'Ariane62 reusable, {n_reuses_list[j]} reuses')
    plt.plot(launches, A64_reusable_cumulative, label=f'Ariane64 reusable, {n_reuses_list[j]} reuses')
plt.plot(launches, A62_expendable_cumulative, label='Ariane62 expendable')
plt.plot(launches, A64_expendable_cumulative, label='Ariane64 expendable')
plt.xlabel('number of launches [-]')
plt.ylabel('Cumulative cost of launches [M€]')
plt.legend()
plt.grid()
plt.show()

plt.suptitle(target_location)
for j in range(len(n_reuses_list)):
    A62_payload_expendable = 10350
    A64_payload_expendable = 21650
    A62_payload_reusable = A62_payload_expendable - payload_mass_penalty
    A64_payload_reusable = A64_payload_expendable - payload_mass_penalty

    A62_expendable_cost_per_kg = []
    A64_expendable_cost_per_kg = []
    A62_reusable_cost_per_kg = []
    A64_reusable_cost_per_kg = []
    for i in range(len(A62_expendable)):
        if account_for_5th_launch_expendable:
            if (i+1)%n_reuses_list[j] == 1:
                A62_payload_reusable = A62_payload_expendable - payload_mass_penalty
                A64_payload_reusable = A64_payload_expendable - payload_mass_penalty
            if (i+1)%n_reuses_list[j] == 0:
                A62_payload_reusable = A62_payload_expendable - payload_mass_penalty + M_p
                A64_payload_reusable = A64_payload_expendable - payload_mass_penalty + M_p
        A62_expendable_cost_per_kg.append((A62_expendable[i] / A62_payload_expendable) * 1000000)
        A64_expendable_cost_per_kg.append((A64_expendable[i] / A64_payload_expendable) * 1000000)
        if A62_payload_reusable > 0.0:
            A62_reusable_cost_per_kg.append((A62_reusable_list[j][i] / A62_payload_reusable) * 1000000)
        if A64_payload_reusable > 0.0:
            A64_reusable_cost_per_kg.append((A64_reusable_list[j][i] / A64_payload_reusable) * 1000000)


    if A62_payload_reusable > 0.0:
        plt.plot(launches, A62_reusable_cost_per_kg, label=f'Ariane62 reusable, {n_reuses_list[j]} reuses')
    if A64_payload_reusable > 0.0:
        plt.plot(launches, A64_reusable_cost_per_kg, label=f'Ariane64 reusable, {n_reuses_list[j]} reuses')
plt.plot(launches, A62_expendable_cost_per_kg, label='Ariane62 expendable')
plt.plot(launches, A64_expendable_cost_per_kg, label='Ariane64 expendable')
plt.xlabel('number of launches [-]')
plt.ylabel('Cost per max kg launched [€/kg]')
plt.legend()
plt.grid()
plt.show()