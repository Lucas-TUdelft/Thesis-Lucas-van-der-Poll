###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import os
import numpy as np
import pickle

import apollo_utils
from plotting_functions import *
from reference_trajectory_selection import *

# Tudatpy imports
import tudatpy
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

# Load spice kernels
spice_interface.load_standard_kernels()

target_location = 'Cabo Verde'
if target_location == 'Paris':
    default_inputs = [7505,
                      np.deg2rad(35.0),
                      np.deg2rad(-0.8),
                      1.0,
                      np.deg2rad(2.0),
                      np.deg2rad((8.0 / (7000 ** 2)))]
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
    # speed = 7375 # m/s
    # heading_angle = np.deg2rad(31.0) # rad
    station_altitude = 0.0  # m
    station_latitude = np.deg2rad(37.7412)  # rad
    station_longitude = np.deg2rad(-25.6756)  # rad
    estimated_flight_time = 750  # s
    # guidance_K = 1 # -
    # deadband_values = [np.deg2rad(2.0), np.deg2rad((8.0 / (7000 ** 2)))] # rad, rad/(m/s^2)

labels = ['Least mp', 'Initial Guess']
trajectory_values = [0.02, 0.05, 0.08]

times = []
altitudes = []
gloads = []
heatfluxes = []
banks = []
latitudes = []
longitudes = []

for i in range(len(labels)):
    print(labels[i])
    '''
    speed = default_inputs[0]
    heading_angle = default_inputs[1]
    flight_path_angle = default_inputs[2]
    guidance_K = default_inputs[3]
    deadband_c0 = trajectory_values[i]  #default_inputs[4]
    deadband_c1 = default_inputs[5]

    
    '''
    '''
    # Natal
    if i == 0:
        speed = 6.42041334e+03
        heading_angle = 2.19759720e+00
        flight_path_angle = -1.14722825e-02
        guidance_K = 2.88804494e+00
        deadband_c0 = 8.24738566e-03
        deadband_c1 = 6.29421765e-10
    if i == 1:
        speed = default_inputs[0]
        heading_angle = default_inputs[1]
        flight_path_angle = default_inputs[2]
        guidance_K = default_inputs[3]
        deadband_c0 = default_inputs[4]  # default_inputs[4]
        deadband_c1 = default_inputs[5]
    if i == 2:
        speed = 6.40756672e+03
        heading_angle = 2.19802973e+00
        flight_path_angle = -1.06078614e-02
        guidance_K = 3.54452788e+00
        deadband_c0 = 2.20202191e-02
        deadband_c1 = 6.44187621e-10
    '''
    # Cabo Verde

    if i == 0:
        speed = 6.93255546e+03
        heading_angle = 1.19658593e+00
        flight_path_angle = -1.13154216e-02
        guidance_K = 1.34361265e+00
        deadband_c0 = 7.41931065e-03
        deadband_c1 = 1.43087627e-09
    if i == 1:
        speed = default_inputs[0]
        heading_angle = default_inputs[1]
        flight_path_angle = default_inputs[2]
        guidance_K = default_inputs[3]
        deadband_c0 = default_inputs[4]  # default_inputs[4]
        deadband_c1 = default_inputs[5]

    deadband_values = [deadband_c0, deadband_c1]

    ###########################################################################
    # DEFINE GLOBAL SETTINGS ##################################################
    ###########################################################################

    # Choose whether output of the propagation is written to files
    write_results_to_file = True
    # Get path of current directory
    current_dir = os.path.dirname(__file__)

    # Set simulation start epoch
    simulation_start_epoch = 0.0  # s
    # Set termination conditions
    maximum_duration = constants.JULIAN_DAY  # s
    termination_altitude = 30.0E3  # m

    # Define settings for celestial bodies
    bodies_to_create = ['Earth', 'Moon', 'Sun']

    # Define Ground station settings
    # target_location = 'Cabo Verde'

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

    # rotation model
    body_settings.get('Earth').rotation_model_settings = environment_setup.rotation_model.gcrs_to_itrs(
        base_frame='J2000')
    body_settings.get('Earth').gravity_field_settings.associated_reference_frame = 'ITRS'

    # create bodies
    bodies = environment_setup.create_system_of_bodies(body_settings)

    # create ground station
    ground_station_settings = environment_setup.ground_station.basic_station(
        "LandingPad",
        [station_altitude, station_latitude, station_longitude],
        element_conversion.geodetic_position_type)
    environment_setup.add_ground_station(bodies.get_body("Earth"), ground_station_settings)

    # capsule
    bodies.create_empty_body('Capsule')
    new_capsule_mass = 10648.25  # kg
    bodies.get_body('Capsule').set_constant_mass(new_capsule_mass)
    reference_area = 60.82  # m^2
    lookup_tables_path = os.path.join(os.getcwd(), "AerodynamicLookupTables")
    aero_coefficients_files = {0: os.path.join(lookup_tables_path, "CD_table.txt"),
                               2: os.path.join(lookup_tables_path, "CL_table.txt")}

    aero_coefficient_settings = environment_setup.aerodynamic_coefficients.tabulated_force_only_from_files(
        force_coefficient_files=aero_coefficients_files,
        reference_area=reference_area,
        independent_variable_names=[environment.altitude_dependent, environment.mach_number_dependent]
    )

    environment_setup.add_aerodynamic_coefficient_interface(bodies, 'Capsule', aero_coefficient_settings)

    # flight conditions
    environment_setup.add_flight_conditions(bodies, 'Capsule', 'Earth')

    # bank angle guidance
    aerodynamic_guidance_object = Util.ApolloGuidance.from_file(
        'apollo_data_vref.npz', bodies, deadband_values, estimated_flight_time, K=guidance_K)
    rotation_model_settings = environment_setup.rotation_model.aerodynamic_angle_based(
        'Earth', '', 'BodyFixed', aerodynamic_guidance_object.getAerodynamicAngles)
    environment_setup.add_rotation_model(bodies, 'Capsule', rotation_model_settings)

    # termination settings
    # Time
    time_termination_settings = propagation_setup.propagator.time_termination(
        simulation_start_epoch + maximum_duration,
        terminate_exactly_on_final_condition=False
    )
    # Altitude
    lower_altitude_termination_settings = propagation_setup.propagator.dependent_variable_termination(
        dependent_variable_settings=propagation_setup.dependent_variable.altitude('Capsule', 'Earth'),
        limit_value=termination_altitude,
        use_as_lower_limit=True,
        terminate_exactly_on_final_condition=False
    )
    # Define list of termination settings
    termination_settings_list = [time_termination_settings,
                                 lower_altitude_termination_settings]
    # Create termination settings object (when either the time of altitude condition is reached: propaation terminates)
    hybrid_termination_settings = propagation_setup.propagator.hybrid_termination(termination_settings_list,
                                                                                  fulfill_single_condition=True)
    # dependent variables
    dependent_variables_to_save = [propagation_setup.dependent_variable.mach_number('Capsule', 'Earth'),
                                   propagation_setup.dependent_variable.altitude('Capsule', 'Earth'),
                                   propagation_setup.dependent_variable.local_aerodynamic_g_load('Capsule',
                                                                                                 'Earth'),
                                   propagation_setup.dependent_variable.keplerian_state('Capsule', 'Earth'),
                                   propagation_setup.dependent_variable.relative_position('Capsule', 'Earth'),
                                   propagation_setup.dependent_variable.relative_velocity('Capsule', 'Earth'),
                                   propagation_setup.dependent_variable.geodetic_latitude('Capsule', 'Earth'),
                                   propagation_setup.dependent_variable.longitude('Capsule', 'Earth'),
                                   propagation_setup.dependent_variable.bank_angle('Capsule', 'Earth'),
                                   propagation_setup.dependent_variable.relative_speed('Capsule', 'Earth'),
                                   propagation_setup.dependent_variable.density('Capsule', 'Earth')
                                   ]

    # body to propagate and central body
    bodies_to_propagate = ['Capsule']
    central_bodies = ['Earth']

    # Define accelerations acting on capsule
    acceleration_settings_on_vehicle = {
        'Earth': [propagation_setup.acceleration.spherical_harmonic_gravity(6, 6),
                  propagation_setup.acceleration.aerodynamic()],
        'Moon': [propagation_setup.acceleration.point_mass_gravity()],
        'Sun': [propagation_setup.acceleration.point_mass_gravity()]
    }
    # Create acceleration models.
    acceleration_settings = {'Capsule': acceleration_settings_on_vehicle}
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies,
        acceleration_settings,
        bodies_to_propagate,
        central_bodies)

    # initial state
    radial_distance = spice_interface.get_average_radius('Earth') + 157.7E3
    latitude = np.deg2rad(5.3)
    longitude = np.deg2rad(-50.0)
    # flight_path_angle = np.deg2rad(-0.8)

    # Convert spherical elements to body-fixed cartesian coordinates
    initial_cartesian_state_body_fixed = element_conversion.spherical_to_cartesian_elementwise(
        radial_distance, latitude, longitude, speed, flight_path_angle, heading_angle)
    # Get rotational ephemerides of the Earth
    earth_rotational_model = bodies.get_body('Earth').rotation_model
    # Transform the state to the global (inertial) frame
    initial_cartesian_state_inertial = environment.transform_to_inertial_orientation(
        initial_cartesian_state_body_fixed,
        simulation_start_epoch,
        earth_rotational_model)

    # pre-maneuver initial state
    radial_distance_disconnect = spice_interface.get_average_radius('Earth') + 157.7E3
    latitude_disconnect = np.deg2rad(5.3)
    longitude_disconnect = np.deg2rad(-50.0)
    flight_path_angle_disconnect = np.deg2rad(-0.8)
    speed_disconnect = 6.93E3
    heading_angle_disconnect = np.deg2rad(95.25)

    # Convert spherical elements to body-fixed cartesian coordinates
    initial_cartesian_state_body_fixed_disconnect = element_conversion.spherical_to_cartesian_elementwise(
        radial_distance_disconnect, latitude_disconnect, longitude_disconnect, speed_disconnect,
        flight_path_angle_disconnect, heading_angle_disconnect)
    # Transform the state to the global (inertial) frame
    initial_cartesian_state_inertial_disconnect = environment.transform_to_inertial_orientation(
        initial_cartesian_state_body_fixed_disconnect,
        simulation_start_epoch,
        earth_rotational_model)

    # propagator and integrator
    propagator_settings = propagation_setup.propagator.translational(central_bodies,
                                                                     acceleration_models,
                                                                     bodies_to_propagate,
                                                                     initial_cartesian_state_inertial,
                                                                     simulation_start_epoch,
                                                                     None,
                                                                     hybrid_termination_settings,
                                                                     propagation_setup.propagator.cowell,
                                                                     output_variables=dependent_variables_to_save)

    step_size = 1.0
    propagator_settings.integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(
        step_size,
        propagation_setup.integrator.CoefficientSets.rkf_56)

    # generate guidance entry conditions
    dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        bodies,
        propagator_settings)

    h0 = aerodynamic_guidance_object.h0
    v0 = aerodynamic_guidance_object.v0
    gamma0 = np.rad2deg(aerodynamic_guidance_object.gamma0)
    t0 = aerodynamic_guidance_object.t0
    s_target = aerodynamic_guidance_object.s_target
    estimated_flight_time = result2array(dynamics_simulator.dependent_variable_history)[:, 0][-1]
    # print('estimated flight time:', estimated_flight_time)

    # acquire reference bank angle and generate corresponding reference trajectory
    bank_initial = [5.0, 5.0, 5.0]
    target_margin = 5000
    max_g_constraint = 10
    max_heatflux_constraint = 1.0 * 10 ** 6
    max_loads = [max_g_constraint, max_heatflux_constraint]
    generate_reference_trajectory_file(h0, v0, gamma0, t0, bank_initial, s_target, target_margin, max_loads)

    # bank angle guidance
    aerodynamic_guidance_object = Util.ApolloGuidance.from_file(
        'apollo_data_vref.npz', bodies, deadband_values, estimated_flight_time, K=guidance_K)
    bodies.get_body('Capsule').rotation_model.reset_aerodynamic_angle_function(
        aerodynamic_guidance_object.getAerodynamicAngles)

    # simulation
    dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        bodies,
        propagator_settings)

    state_history = dynamics_simulator.state_history
    dependent_variables = dynamics_simulator.dependent_variable_history

    state_history_array = result2array(state_history)
    dependent_variables_array = result2array(dependent_variables)

    dependent_variables_time = dependent_variables_array[:, 0]
    h = dependent_variables_array[:, 2]
    velocity_vector = dependent_variables_array[:, 13:16]
    latitude = np.rad2deg(dependent_variables_array[:, 16])
    longitude = np.rad2deg(dependent_variables_array[:, 17])
    bank = np.rad2deg(dependent_variables_array[:, 18])
    vel = dependent_variables_array[:, 19]
    g = dependent_variables_array[:, 3]
    rho = dependent_variables_array[:, 20]

    v_3 = vel ** 3
    k_heatflux = 1.83 * 10 ** (-4)
    R_n = 1.861  # m
    heatflux = k_heatflux * np.sqrt(rho / R_n) * v_3

    Earth_radius = 6371 * 10 ** 3  # m
    final_vehicle_position_bodyfixed = bodies.get_body(
        'Capsule').flight_conditions.body_centered_body_fixed_state[0:3]
    final_vehicle_position_bodyfixed_unit = final_vehicle_position_bodyfixed / np.linalg.norm(
        final_vehicle_position_bodyfixed)
    final_groudstation_position_bodyfixed = bodies.get_body("Earth").get_ground_station(
        "LandingPad").station_state.get_cartesian_position(0.0)
    final_groudstation_position_bodyfixed_unit = final_groudstation_position_bodyfixed / np.linalg.norm(
        final_groudstation_position_bodyfixed)
    dot_product = np.dot(final_vehicle_position_bodyfixed_unit, final_groudstation_position_bodyfixed_unit)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    final_distance_to_target = Earth_radius * np.arccos(dot_product)
    print('final distance to target:', final_distance_to_target)

    # calculate delta-V
    initial_velocity_correction = (initial_cartesian_state_inertial_disconnect -
                                   initial_cartesian_state_inertial)[3:6]
    delta_V = np.linalg.norm(initial_velocity_correction)
    Isp = 360
    g0 = 9.807
    m0 = bodies.get_body('Capsule').mass * np.exp(delta_V / (Isp * g0))
    mp = m0 - bodies.get_body('Capsule').mass
    print('propellant mass:', mp)

    times.append(dependent_variables_time)
    altitudes.append(h)
    gloads.append(g)
    heatfluxes.append(heatflux)
    banks.append(bank)
    latitudes.append(latitude)
    longitudes.append(longitude)



altitude_comparison_plot(altitudes,times,labels)
aero_comparison_plot(gloads,times,labels)
heatflux_comparison_plot(heatfluxes,times,labels)
bank_comparison_plot(banks,times,labels)
latlong_comparison_plot(latitudes,longitudes,
                        np.rad2deg(station_latitude),np.rad2deg(station_longitude),labels)