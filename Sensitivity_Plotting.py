import pickle
import matplotlib.pyplot as plt
import numpy as np
import math
import os

locations = ['Natal', 'Cabo Verde', 'Canarias']
seeds = [42, 22, 96, 35, 11]
uncertainties = [10,8,6,4,2,1,0.5,0.25,0.1]

for location in locations:

    output_folder = 'SimulationOutput'
    output_subfolder = 'sensitivity analysis'
    output_folder = os.path.join(output_folder, output_subfolder)

    filename = location + '.dat'
    filename = os.path.join(output_folder, filename)

    file = open(filename, 'rb')
    data = pickle.load(file)
    file.close()

    simulations_within_range = data[0]
    range_miss_all = data[1]

    print(data)
    for i in range(len(seeds)):
        plt.plot(uncertainties, simulations_within_range[i], label='seed ' + str(seeds[i]))
    plt.suptitle(location)
    plt.xlabel('uncertainty value')
    plt.ylabel('number of simulations within 5km range [-]')
    plt.grid()
    plt.legend()
    plt.show()

    for i in range(len(seeds)):
        plt.plot(uncertainties, range_miss_all[i], label='seed ' + str(seeds[i]))
    plt.suptitle(location)
    plt.xlabel('uncertainty value')
    plt.ylabel('average range miss of simulations outside 5km range [m]')
    plt.grid()
    plt.legend()
    plt.show()

    average_simulations_within_range = []
    average_range_miss = []
    for i in range(len(simulations_within_range[0])):
        sum_sims_i = 0
        sum_range_i = 0
        for j in range(len(simulations_within_range)):
            sum_sims_i = sum_sims_i + simulations_within_range[j][i]
            sum_range_i = sum_range_i + range_miss_all[j][i]

        averaged_sims = sum_sims_i / len(simulations_within_range)
        average_simulations_within_range.append(averaged_sims)
        averaged_range = sum_range_i / len(range_miss_all)
        average_range_miss.append(averaged_range)

    plt.plot(uncertainties, average_simulations_within_range)
    plt.suptitle(location)
    plt.xlabel('uncertainty value')
    plt.ylabel('number of simulations within 5km range [-], for all seeds')
    plt.grid()
    plt.legend()
    plt.show()

    plt.plot(uncertainties, average_simulations_within_range)
    plt.suptitle(location)
    plt.xlabel('uncertainty value')
    plt.ylabel('average range miss of simulations outside 5km range [m], for all seeds')
    plt.grid()
    plt.legend()
    plt.show()


