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
    plt.show()

    plt.plot(uncertainties, average_range_miss)
    plt.suptitle(location)
    plt.xlabel('uncertainty value')
    plt.ylabel('average range miss of simulations outside 5km range [m], for all seeds')
    plt.grid()
    plt.show()

    # [[3, 6, 7, 6, 7, 9, 10, 10, 10], [5, 6, 6, 6, 9, 9, 10, 10, 10], [4, 7, 6, 7, 8, 8, 10, 10, 10], [6, 5, 3, 6, 8, 7, 8, 9, 10], [6, 6, 7, 8, 4, 7, 10, 10, 10]] [[768.3998462179943, 777.8722460656561, 497.9754470798628, 730.097793058284, 265.45961459248093, 169.13905647397314, 0, 0, 0], [1222.0781349954264, 1361.82902394674, 618.4842753207192, 376.94698410097476, 410.2892395338531, 46.77592021021428, 0, 0, 0], [635.8785959514731, 642.0009512922097, 560.9405897777185, 394.6503604701411, 556.6907487010126, 129.2983583007549, 0, 0, 0], [856.7286059150831, 1321.5004567541514, 431.8423573484693, 613.9118796662938, 294.9604698091016, 293.4306266927903, 220.77498161406174, 414.0398449895947, 0], [843.0485500391956, 1232.0261751659204, 604.3455022747366, 212.17051784574915, 130.08647146311446, 150.56415524936588, 0, 0, 0]]


