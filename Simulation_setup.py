# General imports
import numpy as np
import os
import time as tm

# Tudatpy imports
from tudatpy.data import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import environment
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.math import interpolators
from tudatpy.util import result2array
from plotting_functions import *

# Problem-specific imports
import Simulationsetup_utilities as Util

###########################################################################
# DEFINE GLOBAL SETTINGS ##################################################
###########################################################################

# Load spice kernels
spice_interface.load_standard_kernels()
# Using shape parameters for now
shape_parameters = [3.8686343422,
                    2.6697460404,
                    0.6877576649,
                    -0.7652400717,
                    0.3522259173,
                    0.34906585]

# Choose whether benchmark is run
use_benchmark = True
# Choose whether output of the propagation is written to files
write_results_to_file = True
# Get path of current directory
current_dir = os.path.dirname(__file__)

###########################################################################
# DEFINE SIMULATION SETTINGS ##############################################
###########################################################################

# Set simulation start epoch
simulation_start_epoch = 0.0  # s
# Set termination conditions
maximum_duration = constants.JULIAN_DAY  # s
termination_altitude = 30.0E3  # m
# Set vehicle properties
capsule_density = 250.0  # kg m-3

###########################################################################
# CREATE ENVIRONMENT ######################################################
###########################################################################

# Initialize dictionary to save simulation output
simulation_results = dict()

# Set number of models to loop over
number_of_models = 9

# Set the interpolation step at which different runs are compared
output_interpolation_step = 2.0  # s

# Loop over different model settings
for model_test in range(number_of_models):

    # Define settings for celestial bodies
    #bodies_to_create = ['Earth']

    # third-bodies test
    '''
    if model_test == 0:
        bodies_to_create = ['Earth']
    if model_test == 1:
        bodies_to_create = ['Earth', 'Moon']
    if model_test == 2:
        bodies_to_create = ['Earth', 'Sun']
    if model_test == 3:
        bodies_to_create = ['Earth', 'Jupiter']
    if model_test == 4:
        bodies_to_create = ['Earth', 'Mars']
    if model_test == 5:
        bodies_to_create = ['Earth', 'Venus']
    if model_test == 6:
        bodies_to_create = ['Earth', 'Moon', 'Sun']
    '''

    bodies_to_create = ['Earth', 'Moon', 'Sun']

    # Define coordinate system
    global_frame_origin = 'Earth'
    global_frame_orientation = 'J2000'

    # Create body settings
    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create,
        global_frame_origin,
        global_frame_orientation)

    # shape test
    '''
    if model_test == 1:
        equitorial_radius = 6378137.0
        flattening = 1 / 298.25
        body_settings.get('Earth').shape_settings = environment_setup.shape.oblate_spherical(
            equitorial_radius, flattening)
    '''

    # atmosphere test
    '''
    if model_test == 1:
        body_settings.get("Earth").atmosphere_settings = environment_setup.atmosphere.exponential_predefined('Earth')
    if model_test == 2:
        body_settings.get("Earth").atmosphere_settings = environment_setup.atmosphere.nrlmsise00()
    '''

    # rotation test
    '''
    if model_test == 1:
        body_settings.get('Earth').rotation_model_settings = environment_setup.rotation_model.simple_from_spice(
            global_frame_orientation, 'IAU_Earth', 'IAU_Earth', simulation_start_epoch)
    if model_test == 2:
        body_settings.get('Earth').rotation_model_settings = environment_setup.rotation_model.gcrs_to_itrs(
            base_frame='J2000')
    '''
    '''
    # ephemeris test
    if model_test == 1:
        body_settings.get('Earth').ephemeris_settings = environment_setup.ephemeris.keplerian_from_spice(
            'Earth', simulation_start_epoch, spice_interface.get_body_gravitational_parameter('Sun'),
            frame_orientation='J2000')
    if model_test == 2:
        body_settings.get('Moon').ephemeris_settings = environment_setup.ephemeris.keplerian_from_spice(
            'Moon', simulation_start_epoch, spice_interface.get_body_gravitational_parameter('Earth'),
            frame_orientation='J2000')
    if model_test == 3:
        body_settings.get('Sun').ephemeris_settings = environment_setup.ephemeris.keplerian_from_spice(
            'Sun', simulation_start_epoch, spice_interface.get_body_gravitational_parameter('Sun'),
            frame_orientation='J2000')
    if model_test == 4:
        body_settings.get('Earth').ephemeris_settings = environment_setup.ephemeris.keplerian_from_spice(
            'Earth', simulation_start_epoch, spice_interface.get_body_gravitational_parameter('Sun'),
            frame_orientation='J2000')
        body_settings.get('Moon').ephemeris_settings = environment_setup.ephemeris.keplerian_from_spice(
            'Moon', simulation_start_epoch, spice_interface.get_body_gravitational_parameter('Earth'),
            frame_orientation='J2000')
        body_settings.get('Sun').ephemeris_settings = environment_setup.ephemeris.keplerian_from_spice(
            'Sun', simulation_start_epoch, spice_interface.get_body_gravitational_parameter('Sun'),
            frame_orientation='J2000')
    '''
    equitorial_radius = 6378137.0
    flattening = 1 / 298.25
    body_settings.get('Earth').shape_settings = environment_setup.shape.oblate_spherical(
        equitorial_radius, flattening)

    body_settings.get("Earth").atmosphere_settings = environment_setup.atmosphere.nrlmsise00()

    body_settings.get(
        "Earth").gravity_field_settings = environment_setup.gravity_field.predefined_spherical_harmonic(
        environment_setup.gravity_field.ggm02c, 32)

    body_settings.get('Earth').rotation_model_settings = environment_setup.rotation_model.gcrs_to_itrs(
        base_frame='J2000')
    body_settings.get('Earth').gravity_field_settings.associated_reference_frame = 'ITRS'

    # Create bodies
    bodies = environment_setup.create_system_of_bodies(body_settings)

    # add_capsule_to_body_system function
    Util.add_capsule_to_body_system(bodies,
                                    shape_parameters,
                                    capsule_density)

    environment_setup.add_flight_conditions(bodies, 'Capsule', 'Earth')

    constant_angles = np.zeros([3, 1])
    constant_angles[0] = shape_parameters[5]
    angle_function = lambda time: constant_angles
    environment_setup.add_rotation_model(bodies, 'Capsule',
                                         environment_setup.rotation_model.aerodynamic_angle_based(
                                             'Earth', 'J2000', 'CapsuleFixed', angle_function))

    ###########################################################################
    # CREATE (CONSTANT) PROPAGATION SETTINGS ##################################
    ###########################################################################

    # Retrieve termination settings
    termination_settings = Util.get_termination_settings(simulation_start_epoch,
                                                         maximum_duration,
                                                         termination_altitude)
    # Retrieve dependent variables to save
    dependent_variables_to_save = Util.get_dependent_variable_save_settings()
    # Check whether there is any
    are_dependent_variables_to_save = False if not dependent_variables_to_save else True

    '''
    initial_state_correction = np.zeros( 6 )
    if model_test == 2:
        initial_state_correction = np.asarray([-4.07492028e+03, 3.42782237e+03, 2.41498874e+02, -3.86081282e+00,
                                                    -4.67939343e+00, -1.46194781e-02])
    '''

    initial_state_correction = np.asarray([-4.07492028e+03, 3.42782237e+03, 2.41498874e+02, -3.86081282e+00,
                                           -4.67939343e+00, -1.46194781e-02])

    propagator_settings = Util.get_propagator_settings(shape_parameters,
                                                       bodies,
                                                       simulation_start_epoch,
                                                       termination_settings,
                                                       dependent_variables_to_save,
                                                       current_propagator=propagation_setup.propagator.cowell,
                                                       model_choice=model_test,
                                                       initial_state_perturbation=initial_state_correction)

    # Create integrator settings
    propagator_settings.integrator_settings = Util.get_integrator_settings(0, 5, 5, simulation_start_epoch)

    # warmup run
    if model_test == 0:
        dynamics_simulator = numerical_simulation.create_dynamics_simulator(
            bodies, propagator_settings)

    times = []
    for i in range(5):
        t1 = tm.perf_counter()
        dynamics_simulator = numerical_simulation.create_dynamics_simulator(
            bodies, propagator_settings)
        t2 = tm.perf_counter()
        times.append(t2 - t1)

    mean_time = np.mean(times)

    '''
    t1 = tm.perf_counter()

    # Create Shape Optimization Problem object
    dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        bodies, propagator_settings)

    t2 = tm.perf_counter()
    '''

    # Retrieve propagated state and dependent variables
    state_history = dynamics_simulator.state_history
    dependent_variable_history = dynamics_simulator.dependent_variable_history

    print('model number:', model_test)

    if model_test == 0:
        baseline_time = mean_time
        print('baseline time:', baseline_time)
    else:
        new_time = mean_time
        print('new time:', new_time)
        delta_t = new_time - baseline_time
        print('time difference:', delta_t)

    # Save results to a dictionary
    simulation_results[model_test] = [state_history, dependent_variable_history]

    # Get output path
    if model_test == 0:
        subdirectory = '/NominalCase/'
    else:
        subdirectory = '/Model_' + str(model_test) + '/'

    # Decide if output writing is required
    if write_results_to_file:
        output_path = current_dir + subdirectory
    else:
        output_path = None

    # If desired, write output to a file
    if write_results_to_file:
        save2txt(state_history, 'state_history.dat', output_path)
        save2txt(dependent_variable_history, 'dependent_variable_history.dat', output_path)

alt_list = []
aero_list = []
times = []
e_r_list = []
e_r_mag_list = []
interpolation_epochs_list = []
alt_diff_list = []
aero_diff_list = []

# Compare all the model settings with the nominal case
for model_test in range(1, number_of_models):
    # Get output path
    output_path = current_dir + '/Model_' + str(model_test) + '/'

    # Set time limits to avoid numerical issues at the boundaries due to the interpolation
    nominal_state_history = simulation_results[0][0]
    nominal_dependent_variable_history = simulation_results[0][1]
    nominal_times = list(nominal_state_history.keys())

    if model_test == 1:
        nominal_dependent_variable_history_array = result2array(nominal_dependent_variable_history)
        alt = nominal_dependent_variable_history_array[:, 2]
        aero = nominal_dependent_variable_history_array[:, 3]
        alt_list.append(alt)
        aero_list.append(aero)
        times.append(nominal_times)

    # Retrieve current state and dependent variable history
    current_state_history = simulation_results[model_test][0]
    current_dependent_variable_history = simulation_results[model_test][1]
    current_times = list(current_state_history.keys())

    current_dependent_variable_history_array = result2array(current_dependent_variable_history)
    alt = current_dependent_variable_history_array[:, 2]
    aero = current_dependent_variable_history_array[:, 3]
    alt_list.append(alt)
    aero_list.append(aero)
    times.append(current_times)

    # Get limit times at which both histories can be validly interpolated
    interpolation_lower_limit = max(nominal_times[3], current_times[3])
    interpolation_upper_limit = min(nominal_times[-3], current_times[-3])

    # Create vector of epochs to be compared (boundaries are referred to the first case)
    unfiltered_interpolation_epochs = np.arange(current_times[0], current_times[-1], output_interpolation_step)
    unfiltered_interpolation_epochs = [n for n in unfiltered_interpolation_epochs if n <= interpolation_upper_limit]
    interpolation_epochs = [n for n in unfiltered_interpolation_epochs if n >= interpolation_lower_limit]

    # Compare state history
    state_difference_wrt_nominal = Util.compare_models(current_state_history,
                                                       simulation_results[0][0],
                                                       interpolation_epochs,
                                                       output_path,
                                                       'state_difference_wrt_nominal_case.dat')

    # Compare dependent variable history
    dependent_variable_difference_wrt_nominal = Util.compare_models(current_dependent_variable_history,
                                                                    simulation_results[0][1],
                                                                    interpolation_epochs,
                                                                    output_path,
                                                                    'dependent_variable_difference_wrt_nominal_case.dat')

    state_difference_wrt_nominal_array = result2array(state_difference_wrt_nominal)
    e_r = state_difference_wrt_nominal_array[:, 1:4]
    e_r_mag = []
    for i in range(len(e_r)):
        e_r_mag_i = np.sqrt((e_r[i][0] ** 2) + (e_r[i][1] ** 2) + (e_r[i][2] ** 2))
        e_r_mag.append(e_r_mag_i)

    dependent_variable_difference_wrt_nominal_array = result2array(dependent_variable_difference_wrt_nominal)
    alt_diff = dependent_variable_difference_wrt_nominal_array[:, 2]
    aero_diff = dependent_variable_difference_wrt_nominal_array[:, 3]

    e_r_list.append(e_r)
    interpolation_epochs_list.append(interpolation_epochs)
    e_r_mag_list.append(e_r_mag)
    alt_diff_list.append(alt_diff)
    aero_diff_list.append(aero_diff)

# labels
labels_shapes = ['Oblate Sphere']
labels_atmo = ['predefined exponential', 'NRLMSISE']
labels_3rdbodies = ['Moon', 'Sun', 'Jupiter', 'Mars', 'Venus', 'Moon & Sun']
labels_SRP = ['Cannonball, Cp = 1.3', 'Cannonball, Cp = 2.0']
#labels_spherical_harmonics = ['12,12 w.r.t. 10,10']
labels_spherical_harmonics = ['2,0', '2,2', '3,0', '4,0', '4,4', '6,6', '8,8', '10,10']
labels_rotation = ['simple from spice', 'ITRS']
labels_ephemerides = ['Keplerian Earth', 'Keplerian Moon', 'Keplerian Sun', 'All Keplerian']

# Plotting shapes
'''
state_difference_plot(e_r_mag_list, interpolation_epochs_list, labels_shapes)
altitude_comparison_plot(alt_list, times, ['baseline'] + labels_shapes)
aero_comparison_plot(aero_list, times, ['baseline'] + labels_shapes)
'''
# Plotting atmospheric models
'''
state_difference_plot(e_r_mag_list, interpolation_epochs_list, labels_atmo)
altitude_comparison_plot(alt_list, times, ['baseline'] + labels_atmo)
aero_comparison_plot(aero_list, times, ['baseline'] + labels_atmo)
'''
# plotting 3rd bodies
'''
state_difference_plot(e_r_mag_list, interpolation_epochs_list, labels_3rdbodies)
altitude_comparison_plot(alt_list, times, ['baseline'] + labels_3rdbodies)
aero_comparison_plot(aero_list, times, ['baseline'] + labels_3rdbodies)
'''
'''
# plotting SRP
state_difference_plot(e_r_mag_list, interpolation_epochs_list, labels_SRP)
altitude_comparison_plot(alt_list, times, ['baseline'] + labels_SRP)
aero_comparison_plot(aero_list, times, ['baseline'] + labels_SRP)
'''

# plotting Spherical Harmonics
state_difference_plot(e_r_mag_list, interpolation_epochs_list, labels_spherical_harmonics)
altitude_comparison_plot(alt_list, times, ['baseline'] + labels_spherical_harmonics)
aero_comparison_plot(aero_list, times, ['baseline'] + labels_spherical_harmonics)

# plotting rotations
'''
state_difference_plot(e_r_mag_list, interpolation_epochs_list, labels_rotation)
altitude_comparison_plot(alt_list, times, ['baseline'] + labels_rotation)
aero_comparison_plot(aero_list, times, ['baseline'] + labels_rotation)
'''
# plotting ephemerides
'''
state_difference_plot(e_r_mag_list, interpolation_epochs_list, labels_ephemerides)
altitude_comparison_plot(alt_list, times, ['baseline'] + labels_ephemerides)
aero_comparison_plot(aero_list, times, ['baseline'] + labels_ephemerides)
'''
