# General imports
import os
import numpy as np
import pickle
import multiprocessing

import apollo_utils
from plotting_functions import *
from reference_trajectory_selection_multiprocessing import *

# Tudatpy imports
import tudatpy
from tudatpy.data import save2txt
from tudatpy import constants
from tudatpy.interface import spice
from tudatpy.dynamics import environment_setup
from tudatpy.dynamics import propagation_setup
from tudatpy.dynamics import environment
from tudatpy import dynamics.simulator
from tudatpy.astro import element_conversion
#from tudatpy.astro import reference_frames
from tudatpy.kernel.math import interpolators
from tudatpy.util import result2array

import pygmo as pg

# Problem-specific imports
import EntryUtilities_multiprocessing as Util
import OptimisationUtilities as OptUtil

def init_worker():
    spice.load_standard_kernels()

def run_optimization(location):

    try:
        print(f"--- Starting Optimization with Parameter: {location} ---")

        # optimizer names are ihs, nsga2 (multiple of 4), moead, moead_gen, maco, nspso
        # seeds needs to be equal size to num_repeats
        # did: nsga2, maco, moead, moead_gen, ihs, nspso

        optimizer_name = 'moead_gen'
        num_repeats = 2
        num_generations = 20
        num_pops = 400
        seeds = [42, 22, 96, 35, 11]

        target_location = location

        if target_location == 'Paris':
            default_inputs = [7505,
                              np.deg2rad(35.0),
                              np.deg2rad(-0.8),
                              1.0,
                              np.deg2rad(2.0),
                              np.deg2rad((8.0 / (7000 ** 2)))]
            bounds = [[7475.0, 0.59, -0.017, 1.0, 0.005, 0.5 * 10**(-9)],[7540.0, 0.63, -0.010, 11.0, 0.09, 6.0 * 10**(-9)]]
            # speed = 7505 # m/s
            # heading_angle = np.deg2rad(35.0) # rad
            station_altitude = 35.0  # m
            station_latitude = np.deg2rad(48.8575)  # rad
            station_longitude = np.deg2rad(2.3514)  # rad
            estimated_flight_time = 1050  # s
            # guidance_K = 1 # -
            # deadband_values = [np.deg2rad(2.0), np.deg2rad((8.0 / (7000 ** 2)))] # rad, rad/(m/s^2)
        elif target_location == 'Cabo Verde':
            default_inputs = [6970,
                              np.deg2rad(68.5),
                              np.deg2rad(-0.8),
                              1.0,
                              np.deg2rad(2.0),
                              np.deg2rad((8.0 / (7000 ** 2)))]
            bounds = [[6925.0, 1.18, -0.017, 1.0, 0.005, 0.5 * 10**(-9)],[7000.0, 1.21, -0.010, 11.0, 0.09, 6.0 * 10**(-9)]]
            # speed = 6970 # m/s
            # heading_angle = np.deg2rad(68.5) # rad
            station_altitude = 37.0  # m
            station_latitude = np.deg2rad(14.9198)  # rad
            station_longitude = np.deg2rad(-23.5073)  # rad
            estimated_flight_time = 575  # s
            # guidance_K = 1 # -
            # deadband_values = [np.deg2rad(2.0), np.deg2rad((8.0 / (7000 ** 2)))] # rad, rad/(m/s^2)
        elif target_location == 'Natal':
            default_inputs = [6.45E3,
                              np.deg2rad(126),
                              np.deg2rad(-0.8),
                              1.0,
                              np.deg2rad(2.0),
                              np.deg2rad((8.0 / (7000 ** 2)))]
            bounds = [[6400.0, 2.18, -0.017, 1.0, 0.005, 0.5 * 10**(-9)],[6500.0, 2.22, -0.010, 11.0, 0.08, 6.0 * 10**(-9)]]
            # speed = 6.45E3 # m/s
            # heading_angle = np.deg2rad(126) # rad
            station_altitude = 30.0  # m
            station_latitude = np.deg2rad(-5.7842)  # rad
            station_longitude = np.deg2rad(-35.2000)  # rad
            estimated_flight_time = 420  # s
            # guidance_K = 1 # -
            # deadband_values = [np.deg2rad(2.0), np.deg2rad((8.0 / (7000 ** 2)))] # rad, rad/(m/s^2)
        elif target_location == 'Canarias':
            default_inputs = [7275,
                              np.deg2rad(50.5),
                              np.deg2rad(-0.8),
                              1.0,
                              np.deg2rad(2.0),
                              np.deg2rad((8.0 / (7000 ** 2)))]
            bounds = [[7225.0, 0.86, -0.017, 1.0, 0.005, 0.5 * 10**(-9)],[7325.0, 0.90, -0.010, 11.0, 0.08, 6.0 * 10**(-9)]]
            # speed = 7275 # m/s
            # heading_angle = np.deg2rad(50.5) # rad
            station_altitude = 0.0  # m
            station_latitude = np.deg2rad(28.2916)  # rad
            station_longitude = np.deg2rad(-16.6291)  # rad
            estimated_flight_time = 755  # s
            # guidance_K = 1 # -
            # deadband_values = [np.deg2rad(2.0), np.deg2rad((8.0 / (7000 ** 2)))] # rad, rad/(m/s^2)
        elif target_location == 'Azores':
            default_inputs = [7375,
                              np.deg2rad(31.0),
                              np.deg2rad(-0.8),
                              1.0,
                              np.deg2rad(2.0),
                              np.deg2rad((8.0 / (7000 ** 2)))]
            bounds = [[7325, 0.53, -0.016, 1.0, 0.005, 0.5 * 10**(-9)],[7425, 0.56, -0.012, 11.0, 0.08, 6.0 * 10**(-9)]]
            # speed = 7375 # m/s
            # heading_angle = np.deg2rad(31.0) # rad
            station_altitude = 0.0  # m
            station_latitude = np.deg2rad(37.7412)  # rad
            station_longitude = np.deg2rad(-25.6756)  # rad
            estimated_flight_time = 750  # s
            # guidance_K = 1 # -
            # deadband_values = [np.deg2rad(2.0), np.deg2rad((8.0 / (7000 ** 2)))] # rad, rad/(m/s^2)

        opt = OptUtil.optimization(bounds,location,optimizer_name)
        opt.optimize(num_pops, num_generations, num_repeats, seeds)
        output = opt.results
        output_per_generation = opt.results_per_generation

        output_to_store = []

        for i in range(num_repeats):
            x = output[i].get_x()
            y = output[i].get_f()
            y_per_gen = output_per_generation[i]

            output_to_store.append([x,y,y_per_gen])

        output_folder = 'SimulationOutput'
        output_subfolder = 'multi-objective optimization ' + optimizer_name
        output_folder = os.path.join(output_folder, output_subfolder)

        filename = location + '.dat'
        filename = os.path.join(output_folder, filename)
        file = open(filename, 'wb')
        pickle.dump(output_to_store, file)


        print(f"--- Finished Simulation with Parameter: {location} ---")
        return {"param": location, "status": "Success"}

    except Exception as e:
        print(f"Error in sim {location}: {e}")
        return {"param": location, "status": f"Failed: {e}"}



if __name__ == "__main__":

    # Define your 5 different parameter sets
    location_parameters = ['Cabo Verde', 'Natal', 'Canarias', 'Azores', 'Paris']

    # Determine how many cores to use (match your 5 parameters)
    num_cores = 5

    print(f"Launching {num_cores} simulations in parallel...")

    ctx = multiprocessing.get_context("spawn")

    with ctx.Pool(processes=num_cores, initializer=init_worker) as pool:

        results = pool.map(run_optimization, location_parameters)

    print("\nAll simulations complete!")
