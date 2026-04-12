import pickle
import matplotlib.pyplot as plt
import numpy as np
import math
import os

optimizer_names = ['ihs', 'nsga2', 'moead', 'moead_gen', 'maco', 'nspso']
seeds = [42, 22]
'''
fig, axs  = plt.subplots(2,3)

fig.suptitle('Optimization Fitness for Cabo Verde')

for i in range(len(optimizer_names)):
    data_folder = 'SimulationOutput'
    data_subfolder = 'Optimization'
    data_folder = os.path.join(data_folder, data_subfolder)
    optimizer_name = optimizer_names[i]
    data_folder = os.path.join(data_folder, optimizer_name)
    data_file = os.path.join(data_folder, 'Cabo Verde.dat')
    file = open(data_file, 'rb')
    data = pickle.load(file)
    file.close()
    for j in range(len(seeds)):
        results = data[j]
        x = results[0]
        y = results[1]
        axs[math.floor(i/3),i%3].scatter(y[:,0],y[:,1], label = 'seed ' + str(seeds[j]))
    axs[math.floor(i / 3), i % 3].grid()
    axs[math.floor(i / 3), i % 3].set_xlabel('propellant mass fitness')
    axs[math.floor(i / 3), i % 3].set_ylabel('bank-reversal fitness')
    #axs[math.floor(i / 3), i % 3].set_ylim([0, 8])
    #axs[math.floor(i / 3), i % 3].set_xlim([0, 8])
    axs[math.floor(i / 3), i % 3].legend()
    axs[math.floor(i / 3), i % 3].set_title(optimizer_name)
plt.tight_layout()
plt.show()

fig, axs  = plt.subplots(2,3)

fig.suptitle('Optimization Fitness for Natal')

for i in range(len(optimizer_names)):
    data_folder = 'SimulationOutput'
    data_subfolder = 'Optimization'
    data_folder = os.path.join(data_folder, data_subfolder)
    optimizer_name = optimizer_names[i]
    data_folder = os.path.join(data_folder, optimizer_name)
    data_file = os.path.join(data_folder, 'Natal.dat')
    file = open(data_file, 'rb')
    data = pickle.load(file)
    file.close()
    for j in range(len(seeds)):
        results = data[j]
        x = results[0]
        y = results[1]
        axs[math.floor(i/3),i%3].scatter(y[:,0],y[:,1], label = 'seed ' + str(seeds[j]))
    axs[math.floor(i / 3), i % 3].grid()
    axs[math.floor(i / 3), i % 3].set_xlabel('propellant mass fitness')
    axs[math.floor(i / 3), i % 3].set_ylabel('bank-reversal fitness')
    #axs[math.floor(i / 3), i % 3].set_ylim([0, 8])
    #axs[math.floor(i / 3), i % 3].set_xlim([0, 8])
    axs[math.floor(i / 3), i % 3].legend()
    axs[math.floor(i / 3), i % 3].set_title(optimizer_name)
plt.tight_layout()
plt.show()

fig, axs  = plt.subplots(2,3)

fig.suptitle('Optimization Fitness for Azores')

for i in range(len(optimizer_names)):
    data_folder = 'SimulationOutput'
    data_subfolder = 'Optimization'
    data_folder = os.path.join(data_folder, data_subfolder)
    optimizer_name = optimizer_names[i]
    data_folder = os.path.join(data_folder, optimizer_name)
    data_file = os.path.join(data_folder, 'Azores.dat')
    file = open(data_file, 'rb')
    data = pickle.load(file)
    file.close()
    for j in range(len(seeds)):
        results = data[j]
        x = results[0]
        y = results[1]
        axs[math.floor(i/3),i%3].scatter(y[:,0],y[:,1], label = 'seed ' + str(seeds[j]))
    axs[math.floor(i / 3), i % 3].grid()
    axs[math.floor(i / 3), i % 3].set_xlabel('propellant mass fitness')
    axs[math.floor(i / 3), i % 3].set_ylabel('bank-reversal fitness')
    #axs[math.floor(i / 3), i % 3].set_ylim([0, 8])
    #axs[math.floor(i / 3), i % 3].set_xlim([0, 8])
    axs[math.floor(i / 3), i % 3].legend()
    axs[math.floor(i / 3), i % 3].set_title(optimizer_name)
plt.tight_layout()
plt.show()

fig, axs  = plt.subplots(2,3)

fig.suptitle('Optimization Fitness for Canarias')

for i in range(len(optimizer_names)):
    data_folder = 'SimulationOutput'
    data_subfolder = 'Optimization'
    data_folder = os.path.join(data_folder, data_subfolder)
    optimizer_name = optimizer_names[i]
    data_folder = os.path.join(data_folder, optimizer_name)
    data_file = os.path.join(data_folder, 'Canarias.dat')
    file = open(data_file, 'rb')
    data = pickle.load(file)
    file.close()
    for j in range(len(seeds)):
        results = data[j]
        x = results[0]
        y = results[1]
        axs[math.floor(i/3),i%3].scatter(y[:,0],y[:,1], label = 'seed ' + str(seeds[j]))
    axs[math.floor(i / 3), i % 3].grid()
    axs[math.floor(i / 3), i % 3].set_xlabel('propellant mass fitness')
    axs[math.floor(i / 3), i % 3].set_ylabel('bank-reversal fitness')
    #axs[math.floor(i / 3), i % 3].set_ylim([0, 8])
    #axs[math.floor(i / 3), i % 3].set_xlim([0, 8])
    axs[math.floor(i / 3), i % 3].legend()
    axs[math.floor(i / 3), i % 3].set_title(optimizer_name)
plt.tight_layout()
plt.show()

fig, axs  = plt.subplots(2,3)

fig.suptitle('Optimization Fitness for Paris')

for i in range(len(optimizer_names)):
    data_folder = 'SimulationOutput'
    data_subfolder = 'Optimization'
    data_folder = os.path.join(data_folder, data_subfolder)
    optimizer_name = optimizer_names[i]
    data_folder = os.path.join(data_folder, optimizer_name)
    data_file = os.path.join(data_folder, 'Paris.dat')
    file = open(data_file, 'rb')
    data = pickle.load(file)
    file.close()
    for j in range(len(seeds)):
        results = data[j]
        x = results[0]
        y = results[1]
        axs[math.floor(i/3),i%3].scatter(y[:,0],y[:,1], label = 'seed ' + str(seeds[j]))
    axs[math.floor(i / 3), i % 3].grid()
    axs[math.floor(i / 3), i % 3].set_xlabel('propellant mass fitness')
    axs[math.floor(i / 3), i % 3].set_ylabel('bank-reversal fitness')
    #axs[math.floor(i / 3), i % 3].set_ylim([0, 8])
    #axs[math.floor(i / 3), i % 3].set_xlim([0, 8])
    axs[math.floor(i / 3), i % 3].legend()
    axs[math.floor(i / 3), i % 3].set_title(optimizer_name)
plt.tight_layout()
plt.show()
'''
plot_location = 'Paris'
fig, axs  = plt.subplots(2,3)
fig.suptitle('Average fitness during optimization for ' + plot_location)

for i in range(len(optimizer_names)):
    data_folder = 'SimulationOutput'
    data_subfolder = 'Optimization'
    data_folder = os.path.join(data_folder, data_subfolder)
    optimizer_name = optimizer_names[i]
    data_folder = os.path.join(data_folder, optimizer_name)
    data_file = os.path.join(data_folder, plot_location + '.dat')
    file = open(data_file, 'rb')
    data = pickle.load(file)
    file.close()
    for j in range(len(seeds)):
        results = data[j]
        x = results[0]
        y = results[1]
        y_per_gen = results[2]
        y_arr = np.array(y_per_gen)
        avg_arr = np.zeros(y_arr.shape[0])
        for k in range(y_arr.shape[0]):
            gen_avg = 0
            num_valid = 0
            for l in range(y_arr.shape[1]):
                avg_one_pop = np.sum(y_arr[k,l])/2
                if avg_one_pop <= 1000.0:
                    gen_avg += avg_one_pop
                    num_valid += 1
            if num_valid > 0.0:
                avg_arr[k] = gen_avg/num_valid
            else:
                avg_arr[k] = 100.0

        axs[math.floor(i / 3), i % 3].plot(range(len(avg_arr)), avg_arr, label=seeds[j])
    axs[math.floor(i / 3), i % 3].grid()
    axs[math.floor(i / 3), i % 3].set_xlabel('evolutions done')
    axs[math.floor(i / 3), i % 3].set_ylabel('Average fitness')
    axs[math.floor(i / 3), i % 3].legend()
    axs[math.floor(i / 3), i % 3].set_title(optimizer_name)
plt.tight_layout()
plt.show()