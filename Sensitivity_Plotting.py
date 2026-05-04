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
    plt.xlabel('uncertainty value')
    plt.ylabel('average range miss of simulations outside 5km range [m]')
    plt.grid()
    plt.legend()
    plt.show()



