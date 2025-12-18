###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import os
import numpy as np
from plotting_functions import *

# Tudatpy imports
from tudatpy.data import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import environment
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel.math import interpolators
from tudatpy.util import result2array

# Problem-specific imports
import Simulation_setup_utilities as Util

###########################################################################
# DEFINE GLOBAL SETTINGS ##################################################
###########################################################################

# Load spice kernels
spice_interface.load_standard_kernels()

# Choose whether benchmark is run
use_benchmark = True
run_integrator_analysis = True

# Choose whether output of the propagation is written to files
write_results_to_file = True
# Get path of current directory
current_dir = os.path.dirname(__file__)

###########################################################################
# DEFINE SIMULATION SETTINGS ##############################################
###########################################################################

# Using shape parameters for now
shape_parameters = [3.8686343422,
                    2.6697460404,
                    0.6877576649,
                    -0.7652400717,
                    0.3522259173,
                    0.34906585]

capsule_density = 250.0  # kg m-3

# Set simulation start epoch
simulation_start_epoch = 0.0  # s
# Set termination conditions
maximum_duration = constants.JULIAN_DAY  # s
termination_altitude = 30.0E3  # m

###########################################################################
# CREATE ENVIRONMENT ######################################################
###########################################################################

# Define settings for celestial bodies
bodies_to_create = ['Earth']
# Define Ground station settings (Paris)
target_location = 'Paris'
if target_location == 'Paris':
    station_altitude = 35.0 # m
    station_latitude = 48.8575 # deg
    station_longitude = 2.3514 # deg
elif target_location == 'Cabo Verde':
    station_altitude = 37.0 # m
    station_latitude = 14.9198 # deg
    station_longitude = -23.5073 # deg
# Define coordinate system
global_frame_origin = 'Earth'
global_frame_orientation = 'J2000'

# Create body settings
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create,
    global_frame_origin,
    global_frame_orientation)
#Util.add_capsule_settings_to_body_system(body_settings,shape_parameters,capsule_density)
# Create bodies
bodies = environment_setup.create_system_of_bodies(body_settings)
# Create Ground Station
ground_station_settings = environment_setup.ground_station.basic_station(
    "LandingPad",
    [station_altitude, station_latitude, station_longitude],
    element_conversion.geodetic_position_type)
environment_setup.add_ground_station(bodies.get_body("Earth"), ground_station_settings)

# Create and add capsule to body system

Util.add_capsule_to_body_system(bodies,
                                shape_parameters,
                                capsule_density)

# Create rotation model based on aerodynamic guidance
environment_setup.add_flight_conditions(bodies, 'Capsule', 'Earth')

constant_angles = np.zeros([3,1])
constant_angles[ 0 ] = shape_parameters[ 5 ]
angle_function = lambda time : constant_angles
environment_setup.add_rotation_model( bodies, 'Capsule',
                                      environment_setup.rotation_model.aerodynamic_angle_based(
                                          'Earth', 'J2000', 'CapsuleFixed', angle_function ))


###########################################################################
# CREATE PROPAGATION SETTINGS #############################################
###########################################################################

# Retrieve termination settings
termination_settings = Util.get_termination_settings(simulation_start_epoch,
                                                     maximum_duration,
                                                     termination_altitude)

# Retrieve dependent variables to save
dependent_variables_to_save = Util.get_dependent_variable_save_settings()
# Check whether there is any
are_dependent_variables_to_save = False if not dependent_variables_to_save else True

initial_benchmarks = False
investigate_propagators = False
integrator_propagator_analysis = True


###########################################################################
# GENERATE BENCHMARK ######################################################
###########################################################################

if initial_benchmarks:
    benchmark_time_steps = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16]
    # benchmark_time_steps = [2]
    maximum_errors = []
    errors = []
    times = []

    if use_benchmark:
        # Define benchmark interpolator settings to make a comparison between the two benchmarks
        benchmark_interpolator_settings = interpolators.lagrange_interpolation(
            8, boundary_interpolation=interpolators.extrapolate_at_boundary)

        # Create propagator settings for benchmark (Cowell)
        benchmark_propagator_settings = Util.get_propagator_settings(shape_parameters,
                                                                     bodies,
                                                                     simulation_start_epoch,
                                                                     termination_settings,
                                                                     dependent_variables_to_save)
        # Set output path for the benchmarks
        benchmark_output_path = current_dir + '/SimulationOutput/benchmarks/' if write_results_to_file else None

        for i in benchmark_time_steps:
            # Generate benchmarks
            benchmark_time_step = i
            benchmark_list = Util.generate_benchmarks(benchmark_time_step,
                                                      simulation_start_epoch,
                                                      bodies,
                                                      benchmark_propagator_settings,
                                                      are_dependent_variables_to_save,
                                                      benchmark_output_path)

            # Extract benchmark states
            first_benchmark_state_history = benchmark_list[0]
            second_benchmark_state_history = benchmark_list[1]

            # Create state interpolator for first benchmark
            benchmark_state_interpolator = interpolators.create_one_dimensional_vector_interpolator(
                first_benchmark_state_history,
                benchmark_interpolator_settings)

            # Compare benchmark states, returning interpolator of the first benchmark, and writing difference to file if
            # write_results_to_file is set to True
            benchmark_state_difference = Util.compare_benchmarks(first_benchmark_state_history,
                                                                 second_benchmark_state_history,
                                                                 benchmark_output_path,
                                                                 'benchmarks_state_difference.dat')
            benchmark_state_difference_array = result2array(benchmark_state_difference)
            e_r = benchmark_state_difference_array[:, 1:4]
            time = benchmark_state_difference.keys()

            e_r_mag = []
            for j in range(len(e_r)):
                error_magnitude = np.sqrt((e_r[j][0] ** 2) + (e_r[j][1] ** 2) + (e_r[j][2] ** 2))
                e_r_mag.append(error_magnitude)

            e_max = max(e_r_mag)
            maximum_errors.append(e_max)

            '''
            # extract altitude
            benchmark_dependent_variables_array = result2array(benchmark_list[2])
            h = benchmark_dependent_variables_array[:, 2]
            dependent_variables_time = benchmark_list[2].keys()

            altitude_plot(h,dependent_variables_time)


            # extract lat-long plot
            latitude = np.rad2deg(benchmark_dependent_variables_array[:, 16])
            longitude = np.rad2deg(benchmark_dependent_variables_array[:, 17])

            latlong_plot(latitude,longitude,station_latitude,station_longitude)
            '''

        maximum_error_plot(benchmark_time_steps, maximum_errors)

if investigate_propagators:
    ###########################################################################
    # IF DESIRED, GENERATE BENCHMARK ##########################################
    ###########################################################################

    if use_benchmark:
        # Define benchmark interpolator settings to make a comparison between the two benchmarks
        # benchmark_interpolator_settings = interpolators.lagrange_interpolation(
        #    8,boundary_interpolation = interpolators.extrapolate_at_boundary)
        benchmark_interpolator_settings = interpolators.lagrange_interpolation(
            8, boundary_interpolation=interpolators.throw_exception_at_boundary,
            lagrange_boundary_handling=interpolators.LagrangeInterpolatorBoundaryHandling.lagrange_no_boundary_interpolation)

        # Create propagator settings for benchmark (Cowell)
        benchmark_propagator_settings = Util.get_propagator_settings(shape_parameters,
                                                                     bodies,
                                                                     simulation_start_epoch,
                                                                     termination_settings,
                                                                     dependent_variables_to_save)
        # Set output path for the benchmarks
        benchmark_output_path = current_dir + '/SimulationOutput/benchmarks/' if write_results_to_file else None

        # Generate benchmarks
        benchmark_time_step = 0.02
        benchmark_list = Util.generate_benchmarks(benchmark_time_step,
                                                  simulation_start_epoch,
                                                  bodies,
                                                  benchmark_propagator_settings,
                                                  are_dependent_variables_to_save,
                                                  benchmark_output_path)

        # Extract benchmark states (first one is run with benchmark_time_step; second with 2.0*benchmark_time_step)
        first_benchmark_state_history = benchmark_list[0]
        second_benchmark_state_history = benchmark_list[1]
        # Create state interpolator for first benchmark
        benchmark_state_interpolator = interpolators.create_one_dimensional_vector_interpolator(
            first_benchmark_state_history,
            benchmark_interpolator_settings)

        # Compare benchmark states, returning interpolator of the first benchmark, and writing difference to file if
        # write_results_to_file is set to True
        benchmark_state_difference = Util.compare_benchmarks(first_benchmark_state_history,
                                                             second_benchmark_state_history,
                                                             benchmark_output_path,
                                                             'benchmarks_state_difference.dat')

        # Extract benchmark dependent variables, if present
        if are_dependent_variables_to_save:
            first_benchmark_dependent_variable_history = benchmark_list[2]
            second_benchmark_dependent_variable_history = benchmark_list[3]

            # Create dependent variable interpolator for first benchmark
            benchmark_dependent_variable_interpolator = interpolators.create_one_dimensional_vector_interpolator(
                first_benchmark_dependent_variable_history,
                benchmark_interpolator_settings)

            # Compare benchmark dependent variables, returning interpolator of the first benchmark, and writing difference
            # to file if write_results_to_file is set to True
            benchmark_dependent_difference = Util.compare_benchmarks(first_benchmark_dependent_variable_history,
                                                                     second_benchmark_dependent_variable_history,
                                                                     benchmark_output_path,
                                                                     'benchmarks_dependent_variable_difference.dat')

    ###########################################################################
    # RUN SIMULATION FOR VARIOUS SETTINGS #####################################
    ###########################################################################
    """
    Code below propagates states using each propagator (propagator_index=0..6), four multi-stage variable step-size 
    integrators (integrator_index j=0..3) and an RK4 integrator (j=4). For the variable-step integrators, 4 different
    tolerances are used (step_size_index=0..3). For the RK4, 6 different step sizes are used (step_size_index=0..5),
    see use of number_of_integrator_step_size_settings variable. See get_integrator_settings function for more details.

    For each combination of i, j, and k, results are written to directory:
        ShapeOptimization/SimulationOutput/prop_i/int_j/setting_k/

    Specifically:
         state_History.dat                                  Cartesian states as function of time
         dependent_variable_history.dat                     Dependent variables as function of time
         state_difference_wrt_benchmark.dat                 Difference of dependent variables w.r.t. benchmark
         dependent_variable_difference_wrt_benchmark.dat    Difference of states w.r.t. benchmark
         ancillary_simulation_info.dat                      Other information about the propagation (number of function
                                                            evaluations, etc...)

    """

    propagator_errors = []
    propagator_times = []
    unprocessed_states = []

    run_integrator_analysis = True

    if run_integrator_analysis:

        # Define list of propagators
        available_propagators = [propagation_setup.propagator.cowell,
                                 propagation_setup.propagator.encke,
                                 propagation_setup.propagator.gauss_keplerian,
                                 propagation_setup.propagator.gauss_modified_equinoctial,
                                 propagation_setup.propagator.unified_state_model_quaternions,
                                 propagation_setup.propagator.unified_state_model_modified_rodrigues_parameters,
                                 propagation_setup.propagator.unified_state_model_exponential_map]

        propagator_labels = ['Cowell',
                             'Encke',
                             'Kepler',
                             'MEE',
                             'USM7',
                             'USM6',
                             'USM-EM']

        # Define settings to loop over
        number_of_propagators = len(available_propagators)
        # number_of_integrators = 1

        # Loop over propagators
        for propagator_index in range(number_of_propagators):

            # Get current propagator, and define propagation settings
            current_propagator = available_propagators[propagator_index]
            current_propagator_settings = Util.get_propagator_settings(shape_parameters,
                                                                       bodies,
                                                                       simulation_start_epoch,
                                                                       termination_settings,
                                                                       dependent_variables_to_save,
                                                                       current_propagator)

            print('propagator = ' + str(propagator_labels[propagator_index]))

            # Set output path
            output_path = current_dir + '/SimulationOutput/prop_' + str(propagator_index) + '/'

            # Create integrator settings
            integrator_step_size = 8.0
            current_integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(
                integrator_step_size, propagation_setup.integrator.CoefficientSets.rkf_56)
            current_propagator_settings.integrator_settings = current_integrator_settings

            # Create Shape Optimization Problem object
            dynamics_simulator = numerical_simulation.create_dynamics_simulator(
                bodies, current_propagator_settings)

            ### OUTPUT OF THE SIMULATION ###
            # Retrieve propagated state and dependent variables
            state_history = dynamics_simulator.state_history
            unprocessed_state_history = dynamics_simulator.unprocessed_state_history
            dependent_variable_history = dynamics_simulator.dependent_variable_history

            # Get the number of function evaluations (for comparison of different integrators)
            function_evaluation_dict = dynamics_simulator.cumulative_number_of_function_evaluations
            number_of_function_evaluations = list(function_evaluation_dict.values())[-1]
            # Add it to a dictionary
            dict_to_write = {'Number of function evaluations (ignore the line above)': number_of_function_evaluations}
            # Check if the propagation was run successfully
            propagation_outcome = dynamics_simulator.integration_completed_successfully
            dict_to_write['Propagation run successfully'] = propagation_outcome
            # Note if results were written to files
            dict_to_write['Results written to file'] = write_results_to_file
            # Note if benchmark was run
            dict_to_write['Benchmark run'] = use_benchmark
            # Note if dependent variables were present
            dict_to_write['Dependent variables present'] = are_dependent_variables_to_save

            # Save results to a file
            if write_results_to_file:
                save2txt(state_history, 'state_history.dat', output_path)
                save2txt(unprocessed_state_history, 'unprocessed_state_history.dat', output_path)
                save2txt(dependent_variable_history, 'dependent_variable_history.dat', output_path)
                save2txt(dict_to_write, 'ancillary_simulation_info.txt', output_path)
                #environment.save_vehicle_mesh_to_file(
                    #bodies.get_body('Capsule').aerodynamic_coefficient_interface, output_path)

            # Compare the simulation to the benchmarks and write differences to files
            if use_benchmark:
                # Initialize containers
                state_difference = dict()

                # Loop over the propagated states and use the benchmark interpolators
                for epoch in state_history.keys():
                    try:
                        state_difference[epoch] = state_history[epoch] - benchmark_state_interpolator.interpolate(epoch)
                    except:
                        skipped = True

                state_difference_array = result2array(state_difference)
                e_r = state_difference_array[:, 1:4]
                e_r_mag = []
                for j in range(len(e_r)):
                    error_magnitude = np.sqrt((e_r[j][0] ** 2) + (e_r[j][1] ** 2) + (e_r[j][2] ** 2))
                    e_r_mag.append(error_magnitude)
                propagator_errors.append(e_r_mag)

                time = state_difference.keys()
                propagator_times.append(time)

                # Write differences with respect to the benchmarks to files
                if write_results_to_file:
                    save2txt(state_difference, 'state_difference_wrt_benchmark.dat', output_path)

                # Do the same for dependent variables, if present
                if are_dependent_variables_to_save:
                    # Initialize containers
                    dependent_difference = dict()
                    # Loop over the propagated dependent variables and use the benchmark interpolators
                    for epoch in dependent_variable_history.keys():
                        try:
                            dependent_difference[epoch] = dependent_variable_history[
                                                              epoch] - benchmark_dependent_variable_interpolator.interpolate(
                                epoch)
                        except:
                            skipped = True
                    # Write differences with respect to the benchmarks to files
                    if write_results_to_file:
                        save2txt(dependent_difference, 'dependent_variable_difference_wrt_benchmark.dat', output_path)

            unprocessed_state_history_array = result2array(unprocessed_state_history)
            unprocessed_states.append(unprocessed_state_history_array)
            unprocessed_state_time = unprocessed_state_history.keys()
            #unprocessed_state_plot(unprocessed_state_time, unprocessed_state_history_array)

            depependent_variable_array = result2array(dependent_variable_history)
            mach = depependent_variable_array[:, 1]
            altitude = depependent_variable_array[:, 2]
            aero_g = depependent_variable_array[:, 3]
            incl = depependent_variable_array[:, 6]
            r = depependent_variable_array[:, 10:13]
            v = depependent_variable_array[:, 13:16]

            if propagator_labels[propagator_index] == 'Cowell':
                altitude_cowell = altitude
                aero_g_cowell = aero_g
                cowell_time = unprocessed_state_time

            if propagator_labels[propagator_index] == 'USM7':
                altitude_USM7 = altitude
                aero_g_USM7 = aero_g
                USM7_time = unprocessed_state_time

            if propagator_labels[propagator_index] == 'USM6':
                altitude_USM6 = altitude
                aero_g_USM6 = aero_g
                USM6_time = unprocessed_state_time

            if propagator_labels[propagator_index] == 'USM-EM':
                altitude_USM_EM = altitude
                aero_g_USM_EM = aero_g
                USM_EM_time = unprocessed_state_time
                altitudes = [altitude_cowell, altitude_USM7, altitude_USM6, altitude_USM_EM]
                aeros = [aero_g_cowell, aero_g_USM6, aero_g_USM7, aero_g_USM_EM]
                times = [cowell_time, USM6_time, USM7_time, USM_EM_time]
                labels = ['Cowell', 'USM7', 'USM6', 'USM-EM']
                altitude_comparison_plot(altitudes, times, labels)
                aero_comparison_plot(aeros, times, labels)

    propagator_error_plot(propagator_times, propagator_errors, propagator_labels)

if integrator_propagator_analysis:
    ###########################################################################
    # IF DESIRED, GENERATE BENCHMARK ##########################################
    ###########################################################################

    if use_benchmark:
        # Define benchmark interpolator settings to make a comparison between the two benchmarks
        benchmark_interpolator_settings = interpolators.lagrange_interpolation(
            8, boundary_interpolation=interpolators.extrapolate_at_boundary)

        # Create propagator settings for benchmark (Cowell)
        benchmark_propagator_settings = Util.get_propagator_settings(shape_parameters,
                                                                     bodies,
                                                                     simulation_start_epoch,
                                                                     termination_settings,
                                                                     dependent_variables_to_save)
        # Set output path for the benchmarks
        benchmark_output_path = current_dir + '/SimulationOutput/benchmarks/' if write_results_to_file else None

        # Generate benchmarks
        benchmark_time_step = 0.02
        benchmark_list = Util.generate_benchmarks(benchmark_time_step,
                                                  simulation_start_epoch,
                                                  bodies,
                                                  benchmark_propagator_settings,
                                                  are_dependent_variables_to_save,
                                                  benchmark_output_path)

        # Extract benchmark states (first one is run with benchmark_time_step; second with 2.0*benchmark_time_step)
        first_benchmark_state_history = benchmark_list[0]
        second_benchmark_state_history = benchmark_list[1]
        # Create state interpolator for first benchmark
        benchmark_state_interpolator = interpolators.create_one_dimensional_vector_interpolator(
            first_benchmark_state_history,
            benchmark_interpolator_settings)

        # Compare benchmark states, returning interpolator of the first benchmark, and writing difference to file if
        # write_results_to_file is set to True
        benchmark_state_difference = Util.compare_benchmarks(first_benchmark_state_history,
                                                             second_benchmark_state_history,
                                                             benchmark_output_path,
                                                             'benchmarks_state_difference.dat')

        # Extract benchmark dependent variables, if present
        if are_dependent_variables_to_save:
            first_benchmark_dependent_variable_history = benchmark_list[2]
            second_benchmark_dependent_variable_history = benchmark_list[3]

            # Create dependent variable interpolator for first benchmark
            benchmark_dependent_variable_interpolator = interpolators.create_one_dimensional_vector_interpolator(
                first_benchmark_dependent_variable_history,
                benchmark_interpolator_settings)

            # Compare benchmark dependent variables, returning interpolator of the first benchmark, and writing difference
            # to file if write_results_to_file is set to True
            benchmark_dependent_difference = Util.compare_benchmarks(first_benchmark_dependent_variable_history,
                                                                     second_benchmark_dependent_variable_history,
                                                                     benchmark_output_path,
                                                                     'benchmarks_dependent_variable_difference.dat')

    ###########################################################################
    # RUN SIMULATION FOR VARIOUS SETTINGS #####################################
    ###########################################################################
    """
    Code below propagates states using each propagator (propagator_index=0..6), four multi-stage variable step-size 
    integrators (integrator_index j=0..3) and an RK4 integrator (j=4). For the variable-step integrators, 4 different
    tolerances are used (step_size_index=0..3). For the RK4, 6 different step sizes are used (step_size_index=0..5),
    see use of number_of_integrator_step_size_settings variable. See get_integrator_settings function for more details.

    For each combination of i, j, and k, results are written to directory:
        ShapeOptimization/SimulationOutput/prop_i/int_j/setting_k/

    Specifically:
         state_History.dat                                  Cartesian states as function of time
         dependent_variable_history.dat                     Dependent variables as function of time
         state_difference_wrt_benchmark.dat                 Difference of dependent variables w.r.t. benchmark
         dependent_variable_difference_wrt_benchmark.dat    Difference of states w.r.t. benchmark
         ancillary_simulation_info.dat                      Other information about the propagation (number of function
                                                            evaluations, etc...)


    """

    maximum_errors = []
    evaluation_numbers = []

    position_errors = []
    times = []

    if run_integrator_analysis:

        # Define list of propagators
        available_propagators = [propagation_setup.propagator.cowell,
                                 propagation_setup.propagator.encke,
                                 propagation_setup.propagator.gauss_modified_equinoctial]

        # Define settings to loop over
        number_of_propagators = len(available_propagators)
        number_of_integrators = 8

        # Loop over propagators
        for propagator_index in range(number_of_propagators):

            # Get current propagator, and define propagation settings
            current_propagator = available_propagators[propagator_index]
            current_propagator_settings = Util.get_propagator_settings(shape_parameters,
                                                                       bodies,
                                                                       simulation_start_epoch,
                                                                       termination_settings,
                                                                       dependent_variables_to_save,
                                                                       current_propagator)

            maximum_errors_propagator = []
            evaluation_numbers_propagator = []

            position_errors_propagator = []
            times_propagator = []

            # Loop over different integrators
            for integrator_index in range(number_of_integrators):

                number_of_integrator_step_size_settings = 5

                maximum_errors_integrator = []
                evaluation_numbers_integrator = []

                position_errors_integrator = []
                times_integrator = []

                # Loop over all tolerances / step sizes
                for step_size_index in range(number_of_integrator_step_size_settings):
                    # time1 = tm.time()
                    # Print status
                    to_print = 'Current run: \n propagator_index = ' + str(propagator_index) + \
                               '\n integrator_index = ' + str(integrator_index) \
                               + '\n step_size_index = ' + str(step_size_index)
                    print(to_print)
                    # Set output path
                    output_path = current_dir + '/SimulationOutput/prop_' + str(propagator_index) + \
                                  '/int_' + str(integrator_index) + '/step_size_' + str(step_size_index) + '/'

                    # Create integrator settings
                    current_integrator_settings = Util.get_integrator_settings(propagator_index,
                                                                               integrator_index,
                                                                               step_size_index,
                                                                               simulation_start_epoch)
                    current_propagator_settings.integrator_settings = current_integrator_settings
                    # Create Shape Optimization Problem object
                    dynamics_simulator = numerical_simulation.create_dynamics_simulator(
                        bodies, current_propagator_settings)

                    ### OUTPUT OF THE SIMULATION ###
                    # Retrieve propagated state and dependent variables
                    state_history = dynamics_simulator.state_history
                    unprocessed_state_history = dynamics_simulator.unprocessed_state_history
                    dependent_variable_history = dynamics_simulator.dependent_variable_history
                    '''
                    time2 = tm.time()
                    elapsed_time = time2 - time1
                    print(elapsed_time)
                    '''
                    # Get the number of function evaluations (for comparison of different integrators)
                    function_evaluation_dict = dynamics_simulator.cumulative_number_of_function_evaluations
                    number_of_function_evaluations = list(function_evaluation_dict.values())[-1]
                    # Add it to a dictionary
                    dict_to_write = {
                        'Number of function evaluations (ignore the line above)': number_of_function_evaluations}
                    # Check if the propagation was run successfully
                    propagation_outcome = dynamics_simulator.integration_completed_successfully
                    dict_to_write['Propagation run successfully'] = propagation_outcome
                    # Note if results were written to files
                    dict_to_write['Results written to file'] = write_results_to_file
                    # Note if benchmark was run
                    dict_to_write['Benchmark run'] = use_benchmark
                    # Note if dependent variables were present
                    dict_to_write['Dependent variables present'] = are_dependent_variables_to_save

                    # Save results to a file
                    if write_results_to_file:
                        save2txt(state_history, 'state_history.dat', output_path)
                        save2txt(unprocessed_state_history, 'unprocessed_state_history.dat', output_path)
                        save2txt(dependent_variable_history, 'dependent_variable_history.dat', output_path)
                        save2txt(dict_to_write, 'ancillary_simulation_info.txt', output_path)
                        #environment.save_vehicle_mesh_to_file(
                            #bodies.get_body('Capsule').aerodynamic_coefficient_interface, output_path)

                    # Compare the simulation to the benchmarks and write differences to files
                    if use_benchmark:
                        # Initialize containers
                        state_difference = dict()

                        # Loop over the propagated states and use the benchmark interpolators
                        # benchmark states (or dependent variables), producing a warning. Be aware of it!
                        for epoch in state_history.keys():
                            try:
                                state_difference[epoch] = state_history[
                                                              epoch] - benchmark_state_interpolator.interpolate(
                                    epoch)
                            except:
                                skipped = True

                        state_difference_array = result2array(state_difference)
                        state_difference_time = state_difference.keys()
                        e_r = state_difference_array[:, 1:4]
                        e_r_mag = []
                        for j in range(len(e_r)):
                            error_magnitude = np.sqrt((e_r[j][0] ** 2) + (e_r[j][1] ** 2) + (e_r[j][2] ** 2))
                            e_r_mag.append(error_magnitude)

                        e_max = max(e_r_mag)
                        maximum_errors_integrator.append(e_max)
                        evaluation_numbers_integrator.append(number_of_function_evaluations)

                        position_errors_integrator.append(e_r_mag)
                        times_integrator.append(state_difference_time)

                        '''
                        if propagator_index == 0:
                            if integrator_index == 1:
                                if step_size_index == 1:

                                    time_steps = []
                                    time_list = [t for t in state_difference_time]

                                    depependent_variable_array = result2array(dependent_variable_history)
                                    mach = depependent_variable_array[:, 1]
                                    altitude = depependent_variable_array[:, 2]
                                    aero_g = depependent_variable_array[:, 3]

                                    for i in range(len(time_list)):
                                        if i == 0:
                                            delta_t = 1.0
                                        else:
                                            delta_t = time_list[i] - time_list[i - 1]
                                        time_steps.append(delta_t)
                        '''

                        # Write differences with respect to the benchmarks to files
                        if write_results_to_file:
                            save2txt(state_difference, 'state_difference_wrt_benchmark.dat', output_path)

                        # Do the same for dependent variables, if present
                        if are_dependent_variables_to_save:
                            # Initialize containers
                            dependent_difference = dict()
                            # Loop over the propagated dependent variables and use the benchmark interpolators
                            for epoch in dependent_variable_history.keys():
                                try:
                                    dependent_difference[epoch] = dependent_variable_history[
                                                                      epoch] - benchmark_dependent_variable_interpolator.interpolate(
                                        epoch)
                                except:
                                    skipped = True
                                # Write differences with respect to the benchmarks to files
                            if write_results_to_file:
                                save2txt(dependent_difference, 'dependent_variable_difference_wrt_benchmark.dat',
                                         output_path)

                maximum_errors_propagator.append(maximum_errors_integrator)
                evaluation_numbers_propagator.append(evaluation_numbers_integrator)

                position_errors_propagator.append(position_errors_integrator)
                times_propagator.append(times_integrator)

            maximum_errors.append(maximum_errors_propagator)
            evaluation_numbers.append(evaluation_numbers_propagator)

            position_errors.append(position_errors_propagator)
            times.append(times_propagator)

        # Print the ancillary information
        print('\n### ANCILLARY SIMULATION INFORMATION ###')
        for (elem, (info, result)) in enumerate(dict_to_write.items()):
            if elem > 1:
                print(info + ': ' + str(result))

    labels = [['Cowell, RK4(5) variable step', 'Cowell, RK5(6) variable step', 'Cowell, RKDP7(8) variable step',
               'Cowell, RKF12(10) variable step', 'Cowell, RK4(5) fixed step', 'Cowell, RK5(6) fixed step',
               'Cowell, RKDP7(8) fixed step', 'Cowell, RKF12(10) fixed step'],
              ['Encke, RK4(5) variable step', 'Encke, RK5(6) variable step', 'Encke, RKDP7(8) variable step',
               'Encke, RKF12(10) variable step', 'Encke, RK4(5) fixed step', 'Encke, RK5(6) fixed step',
               'Encke, RKDP7(8) fixed step', 'Encke, RKF12(10) fixed step'],
              ['MEE, RK4(5) variable step', 'MEE, RK5(6) variable step', 'MEE, RKDP7(8) variable step',
               'MEE, RKF12(10) variable step', 'MEE, RK4(5) fixed step', 'MEE, RK5(6) fixed step',
               'MEE, RKDP7(8) fixed step', 'MEE, RKF12(10) fixed step']]

    for i in range(len(evaluation_numbers)):
        integrator_propagator_plot(evaluation_numbers[i], maximum_errors[i], labels[i])

    #error_plot([times[0][1][1]], [position_errors[0][1][1]],
    #           ['10^-11, Cowell RK5(6)'])

    #time_steps_plot(time_steps, time_list)
    #altitude_plot(altitude, time_list)
    #Mach_plot(mach, time_list)
    #aero_g_plot(aero_g, time_list)

#maximum_error_plot(benchmark_time_steps,maximum_errors)
#g = environment.SystemOfBodies.get_body('Capsule').mass
#ground_station = bodies.get_body("Earth").get_ground_station("LandingPad")
#ground_station_state = ground_station.station_state.get_cartesian_position(benchmark_dependent_variables_array[:, 0][-1])