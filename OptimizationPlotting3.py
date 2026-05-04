import pickle
import matplotlib.pyplot as plt
import numpy as np
import math
import os

plot_location = 'Paris'
#seeds = [42, 22, 96, 35, 11]
seeds = [42]

fig, axs = plt.subplots(1,1)

for i in range(len(seeds)):
    data_folder = 'SimulationOutput'
    data_subfolder = 'Optimization expanded'
    data_folder = os.path.join(data_folder, data_subfolder)
    data_folder = os.path.join(data_folder, plot_location)
    data_file = os.path.join(data_folder, plot_location + str(seeds[i]) + '.dat')
    file = open(data_file, 'rb')
    data = pickle.load(file)
    file.close()
    x = data[0][0]
    y = data[0][1]
    y_per_gen = data[0][2]
    axs.scatter(y[:,0],y[:,1], label = 'seed ' + str(seeds[i]), s = 10)
    axs.grid()
    axs.set_xlabel('propellant mass fitness')
    axs.set_ylabel('bank-reversal fitness')
    # axs[math.floor(i / 3), i % 3].set_ylim([0, 8])
    # axs[math.floor(i / 3), i % 3].set_xlim([0, 8])
    axs.legend()
    axs.set_title(plot_location)
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1,1)

for i in range(len(seeds)):
    data_folder = 'SimulationOutput'
    data_subfolder = 'Optimization expanded'
    data_folder = os.path.join(data_folder, data_subfolder)
    data_folder = os.path.join(data_folder, plot_location)
    data_file = os.path.join(data_folder, plot_location + str(seeds[i]) + '.dat')
    file = open(data_file, 'rb')
    data = pickle.load(file)
    file.close()
    x = data[0][0]
    y = data[0][1]
    y_per_gen = data[0][2]
    y_arr = np.array(y_per_gen)
    avg_arr = np.zeros(y_arr.shape[0])
    for j in range(y_arr.shape[0]):
        gen_avg = 0
        num_valid = 0
        for k in range(y_arr.shape[1]):
            avg_one_pop = np.sum(y_arr[j, k])

            if avg_one_pop <= 1000.0:
                gen_avg += avg_one_pop
                num_valid += 1
        if num_valid > 0.0:
            avg_arr[j] = gen_avg / num_valid
        else:
            avg_arr[j] = 100.0

    axs.plot(range(len(avg_arr)), avg_arr, label=seeds[i])
    axs.grid()
    axs.set_xlabel('evolutions done')
    axs.set_ylabel('Average fitness')
    axs.legend()
    axs.set_title(plot_location)
plt.tight_layout()
plt.show()

for i in range(len(seeds)):
    data_folder = 'SimulationOutput'
    data_subfolder = 'Optimization expanded'
    data_folder = os.path.join(data_folder, data_subfolder)
    data_folder = os.path.join(data_folder, plot_location)
    data_file = os.path.join(data_folder, plot_location + str(seeds[i]) + '.dat')
    file = open(data_file, 'rb')
    data = pickle.load(file)
    file.close()
    x = data[0][0]
    y = data[0][1]
    y_per_gen = data[0][2]

    best_mp_fitness = 10
    best_bank_fitness = 1.0
    best_mp_fitness_bank = 10
    for i in range(len(y)):
        mp_fitness = y[i,0]
        bank_fitness = y[i,1]

        if mp_fitness < best_mp_fitness:
            best_mp_fitness = mp_fitness
            best_mp_fitness_i = i

        if bank_fitness <= best_bank_fitness:
            if mp_fitness < best_mp_fitness_bank:
                best_mp_fitness_bank = mp_fitness
                best_bank_fitness = bank_fitness
                best_bank_fitness_i = i
                print(i)

    print(best_mp_fitness, best_bank_fitness)
    best_mp_inputs = x[best_mp_fitness_i]
    best_bank_inputs = x[best_bank_fitness_i]
    print(best_mp_inputs, best_bank_inputs)

fig, axs = plt.subplots(1,1)

for i in range(len(seeds)):
    data_folder = 'SimulationOutput'
    data_subfolder = 'Optimization expanded'
    data_folder = os.path.join(data_folder, data_subfolder)
    data_folder = os.path.join(data_folder, plot_location)
    data_file = os.path.join(data_folder, plot_location + str(seeds[i]) + '.dat')
    file = open(data_file, 'rb')
    data = pickle.load(file)
    file.close()
    x = data[0][0]
    y = data[0][1]
    y_per_gen = data[0][2]
    y_arr = np.array(y_per_gen)
    best_fitness_list = []
    for j in range(y_arr.shape[0]):
        best_mp_fitness = 100
        for k in range(y_arr.shape[1]):
            if y_arr[j, k, 0] <= best_mp_fitness:
                best_mp_fitness = y_arr[j, k, 0]
        best_fitness_list.append(best_mp_fitness)
    axs.plot(range(len(avg_arr)), best_fitness_list, label=seeds[i])
    axs.grid()
    axs.set_xlabel('evolutions done')
    axs.set_ylabel('best fitness')
    axs.legend()
    axs.set_title(plot_location)
plt.tight_layout()
plt.show()