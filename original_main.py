###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import os
import numpy as np

import apollo_utils
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
import EntryUtilities as Util

###########################################################################
# DEFINE GLOBAL SETTINGS ##################################################
###########################################################################

# Load spice kernels
spice_interface.load_standard_kernels()

# Choose whether benchmark is run
use_benchmark = True
run_integrator_analysis = False

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
                    0.2548030601]

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
bodies_to_create = ['Earth', 'Moon', 'Sun']
# Define Ground station settings (Paris)
target_location = 'Cabo Verde'
if target_location == 'Paris':
    station_altitude = 35.0 # m
    station_latitude = 48.8575 # deg
    station_longitude = 2.3514 # deg
elif target_location == 'Cabo Verde':
    station_altitude = 37.0 # m
    station_latitude = 14.9198 # deg
    station_longitude = -23.5073 # deg
elif target_location == 'Natal':
    station_altitude = 30.0  # m
    station_latitude = -5.7842  # deg
    station_longitude = -35.2000  # deg
# Define coordinate system
global_frame_origin = 'Earth'
global_frame_orientation = 'J2000'

# Create body settings
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create,
    global_frame_origin,
    global_frame_orientation)

# Earth shape
equitorial_radius = 6378137.0
flattening = 1 / 298.25
body_settings.get('Earth').shape_settings = environment_setup.shape.oblate_spherical(
    equitorial_radius, flattening)

# atmosphere
body_settings.get("Earth").atmosphere_settings = environment_setup.atmosphere.nrlmsise00()

# spherical harmonic field
body_settings.get(
        "Earth").gravity_field_settings = environment_setup.gravity_field.predefined_spherical_harmonic(
        environment_setup.gravity_field.ggm02c, 32)

# keplerian ephemerides
body_settings.get('Earth').ephemeris_settings = environment_setup.ephemeris.keplerian_from_spice(
    'Earth', simulation_start_epoch, spice_interface.get_body_gravitational_parameter('Sun'),
    frame_orientation='J2000')
body_settings.get('Moon').ephemeris_settings = environment_setup.ephemeris.keplerian_from_spice(
    'Moon', simulation_start_epoch, spice_interface.get_body_gravitational_parameter('Earth'),
    frame_orientation='J2000')
body_settings.get('Sun').ephemeris_settings = environment_setup.ephemeris.keplerian_from_spice(
    'Sun', simulation_start_epoch, spice_interface.get_body_gravitational_parameter('Sun'),
    frame_orientation='J2000')

body_settings.get('Earth').rotation_model_settings = environment_setup.rotation_model.gcrs_to_itrs(
        base_frame='J2000')
body_settings.get('Earth').gravity_field_settings.associated_reference_frame = 'ITRS'

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

#aerodynamic_guidance_object = Util.PREDGUID(bodies)
aerodynamic_guidance_object = Util.ApolloGuidance.from_file('apollo_data_vref.npz', bodies, K=1)
#aerodynamic_guidance_object = Util.validation_guidance(bodies)
rotation_model_settings = environment_setup.rotation_model.aerodynamic_angle_based(
    'Earth', '', 'BodyFixed', aerodynamic_guidance_object.getAerodynamicAngles )
environment_setup.add_rotation_model( bodies, 'Capsule', rotation_model_settings )
#print("[TEST] Manual guidance eval at t=0: ", aerodynamic_guidance_object.getAerodynamicAngles(0.0))

'''
STS_guidance_object = Util.STSAerodynamicGuidance(bodies)
rotation_model_settings = environment_setup.rotation_model.aerodynamic_angle_based(
    'Earth', '', 'BodyFixed', STS_guidance_object.getAerodynamicAngles )
environment_setup.add_rotation_model( bodies, 'Capsule', rotation_model_settings )
'''
#angles = aerodynamic_guidance_object.getAerodynamicAngles(0.0)
#print("[TEST] Manual aerodynamic angles at t=0:", angles)

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

###########################################################################
# GENERATE BENCHMARK ######################################################
###########################################################################

#benchmark_time_steps = [0.05,0.1,0.25,0.5,1,2,4,8,16]
benchmark_time_steps = [2]
maximum_errors = []
errors = []
times = []

if use_benchmark:
    # Define benchmark interpolator settings to make a comparison between the two benchmarks
    benchmark_interpolator_settings = interpolators.lagrange_interpolation(
        8,boundary_interpolation = interpolators.extrapolate_at_boundary)

    # Create propagator settings for benchmark (Cowell)
    benchmark_propagator_settings = Util.get_propagator_settings(shape_parameters,
                                                                 bodies,
                                                                 simulation_start_epoch,
                                                                 termination_settings,
                                                                 dependent_variables_to_save )
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

        # extract altitude
        benchmark_dependent_variables_array = result2array(benchmark_list[2])
        h = benchmark_dependent_variables_array[:, 2]
        dependent_variables_time = benchmark_list[2].keys()

        altitude_plot(h,dependent_variables_time)

        # extract lat-long plot
        latitude = np.rad2deg(benchmark_dependent_variables_array[:, 16])
        longitude = np.rad2deg(benchmark_dependent_variables_array[:, 17])

        latlong_plot(latitude,longitude,station_latitude,station_longitude)

        # extract bank angle
        bank_angle = np.rad2deg(benchmark_dependent_variables_array[:, 18])

        bank_plot(bank_angle, dependent_variables_time)


#maximum_error_plot(benchmark_time_steps,maximum_errors)
#g = environment.SystemOfBodies.get_body('Capsule').mass
ground_station = bodies.get_body("Earth").get_ground_station("LandingPad")
ground_station_state = ground_station.station_state.get_cartesian_position(benchmark_dependent_variables_array[:, 0][-1])

#print(bodies.get_body("Capsule").flight_conditions.body_centered_body_fixed_state)

print(ground_station_state)