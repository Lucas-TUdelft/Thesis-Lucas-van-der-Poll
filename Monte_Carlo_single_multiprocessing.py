###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import os
import numpy as np
import pickle
import multiprocessing

import apollo_utils
from plotting_functions import *
from reference_trajectory_selection import *

# Tudatpy imports
import tudatpy
from tudatpy.data import save2txt
from tudatpy import constants
from tudatpy.interface import spice
from tudatpy.numerical_simulation import environment_setup
from tudatpy.numerical_simulation import propagation_setup
from tudatpy.numerical_simulation import environment
from tudatpy import numerical_simulation
from tudatpy.astro import element_conversion
from tudatpy.kernel.math import interpolators
from tudatpy.util import result2array

# Problem-specific imports
import EntryUtilities_multiprocessing as Util

def run_simulation(location):

    try:
        print(f"--- Starting Simulation with Parameter: {location} ---")

        ###########################################################################
        # PREPARE MONTE CARLO ANALYSIS ############################################
        ###########################################################################
        '''
        Monte Carlo input parameters: 
        Target Location
        Initial Velocity, 
        Initial Heading Angle, 
        Initial Flight Path Angle
        Guidance Gain
        Deadband Value c0
        Deadband Value c1
        '''

        parameternames = ['Initial Velocity',
                          'Initial Heading Angle',
                          'Initial Flight Path Angle',
                          'Guidance Gain',
                          'Deadband Value c0',
                          'Deadband Value c1']
        parameternames_axis = ['Initial Velocity [m/s]',
                               'Initial Heading Angle [rad]',
                               'Initial Flight Path Angle [rad]',
                               'Guidance Gain [-]',
                               'Deadband Value c0 [rad]',
                               'Deadband Value c1 [rad/(m/s)^2]']
        # 'Cabo Verde', 'Natal', 'Canarias', 'Azores', 'Paris'
        variation_range_per_parameter = [[str(location)],
                                         [-50.0, 50.0],
                                         [-np.deg2rad(1.0), np.deg2rad(1.0)],
                                         [-np.deg2rad(0.2), np.deg2rad(0.2)],
                                         [0, 10],
                                         [np.deg2rad(-1.5), np.deg2rad(3.0)],
                                         [-np.deg2rad((8.0 / (7000 ** 2))), np.deg2rad((8.0 / (7000 ** 2)))]]

        num_parameters = len(variation_range_per_parameter) - 1
        num_simulations = 20

        np.random.seed(42)

        inputs = np.empty((len(variation_range_per_parameter[0]), num_parameters, num_simulations), dtype=object)
        objectives = np.empty((len(variation_range_per_parameter[0]), num_parameters, num_simulations), dtype=object)
        constraints = np.empty((len(variation_range_per_parameter[0]), num_parameters, num_simulations), dtype=object)

        ###########################################################################
        # PERFORM MONTE CARLO ANALYSIS ############################################
        ###########################################################################

        output_folder = 'SimulationOutput'
        output_subfolder = 'MC single'
        output_folder = os.path.join(output_folder, output_subfolder)

        # Load spice kernels
        spice.load_standard_kernels()

        for i in range(len(variation_range_per_parameter[0])):
            # print(variation_range_per_parameter[0][i])
            target_location = variation_range_per_parameter[0][i]

            for j in range(num_parameters):
                variation = np.random.uniform(variation_range_per_parameter[j + 1][0],
                                              variation_range_per_parameter[j + 1][1],
                                              num_simulations)

                for k in range(num_simulations):
                    #print('target location:', target_location, 'parameter index varied:', j, 'variation', k)

                    # get default input parameters
                    # InitialVelocity,
                    # Initial Heading Angle,
                    # Initial Flight Path Angle
                    # Guidance Gain
                    # Deadband Value c0
                    # Deadband Value c1
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

                    # modify input parameters
                    default_inputs[j] = default_inputs[j] + variation[k]

                    speed = default_inputs[0]
                    heading_angle = default_inputs[1]
                    flight_path_angle = default_inputs[2]
                    guidance_K = default_inputs[3]
                    deadband_c0 = default_inputs[4]
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
                        'Earth', simulation_start_epoch, spice.get_body_gravitational_parameter('Sun'),
                        frame_orientation='J2000')
                    body_settings.get('Moon').ephemeris_settings = environment_setup.ephemeris.keplerian_from_spice(
                        'Moon', simulation_start_epoch, spice.get_body_gravitational_parameter('Earth'),
                        frame_orientation='J2000')
                    body_settings.get('Sun').ephemeris_settings = environment_setup.ephemeris.keplerian_from_spice(
                        'Sun', simulation_start_epoch, spice.get_body_gravitational_parameter('Sun'),
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
                    bodies.get_body('Capsule').mass = new_capsule_mass
                    reference_area = 60.82  # m^2
                    lookup_tables_path = os.path.join(os.getcwd(), "AerodynamicLookupTables")
                    aero_coefficients_files = {0: os.path.join(lookup_tables_path, "CD_table.txt"),
                                               2: os.path.join(lookup_tables_path, "CL_table.txt")}

                    aerodynamics = environment_setup.aerodynamic_coefficients

                    aero_coefficient_settings = environment_setup.aerodynamic_coefficients.tabulated_force_only_from_files(
                        force_coefficient_files=aero_coefficients_files,
                        reference_area=reference_area,
                        independent_variable_names=[aerodynamics.altitude_dependent, aerodynamics.mach_number_dependent]
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
                                                   propagation_setup.dependent_variable.relative_position('Capsule',
                                                                                                          'Earth'),
                                                   propagation_setup.dependent_variable.relative_velocity('Capsule',
                                                                                                          'Earth'),
                                                   propagation_setup.dependent_variable.geodetic_latitude('Capsule',
                                                                                                          'Earth'),
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
                    radial_distance = spice.get_average_radius('Earth') + 157.7E3
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
                    radial_distance_disconnect = spice.get_average_radius('Earth') + 157.7E3
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

                    # optimisation constraints: max g-load, max heatflux, total heatload, final velocity
                    max_gload = max(g)
                    v_3 = vel ** 3
                    k_heatflux = 1.83 * 10 ** (-4)
                    R_n = 1.861  # m
                    heatflux = k_heatflux * np.sqrt(rho / R_n) * v_3
                    max_heatflux = max(heatflux)
                    total_heatload = np.trapz(heatflux)
                    final_groudstation_position_bodyfixed = bodies.get_body("Earth").get_ground_station(
                        "LandingPad").station_state.get_cartesian_position(0.0)
                    station_final_intertial_velocity = environment.transform_to_inertial_orientation(
                        np.append(final_groudstation_position_bodyfixed, [0.0, 0.0, 0.0]),
                        dependent_variables_time[-1],
                        bodies.get_body('Earth').rotation_model
                    )[3:6]
                    final_velocity = np.linalg.norm(station_final_intertial_velocity - velocity_vector[-1])
                    succesfull_completion = dynamics_simulator.integration_completed_successfully

                    '''
                    print(dependent_variables_time[-1])
                    print('optimisation constraints:')
                    print('max g-load:', max_gload, 'g')
                    print('max heatflux:', max_heatflux / 1e3, 'kW/m^2')
                    print('total heat load:', total_heatload / 1e6, 'MJ/m^2')
                    print('final velocity:', final_velocity, 'm/s')
                    '''

                    # optimisation objectives: final distance to target, propellant, number of bank reversals
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

                    # calculate delta-V
                    # print(initial_cartesian_state_inertial_disconnect, initial_cartesian_state_inertial,
                    #      initial_cartesian_state_inertial_disconnect - initial_cartesian_state_inertial)
                    initial_velocity_correction = (initial_cartesian_state_inertial_disconnect -
                                                   initial_cartesian_state_inertial)[3:6]
                    delta_V = np.linalg.norm(initial_velocity_correction)
                    Isp = 360
                    g0 = 9.807
                    m0 = new_capsule_mass * np.exp(delta_V / (Isp * g0))
                    mp = m0 - new_capsule_mass

                    n_bank_reversals = aerodynamic_guidance_object.number_of_bank_reversals

                    '''
                    print('optimisation objectives:')
                    print('final distance to target:', final_distance_to_target / 1e3, 'km')
                    print('propellant mass for initial state correction:', mp, 'kg')
                    '''

                    inputs[i][j][k] = default_inputs[j]
                    objectives[i][j][k] = [final_distance_to_target, mp, n_bank_reversals]
                    constraints[i][j][k] = [max_gload, max_heatflux, total_heatload, final_velocity, succesfull_completion]

        # constraint values
        max_total_heatload_constraint = 200e6  # J/m^2
        max_final_velocity_constraint = 900  # m/s

        data_path = os.path.join(output_folder, ' MC single ' + variation_range_per_parameter[0][i] + '.dat')
        savedata = [inputs, objectives, constraints]
        with open(data_path, 'wb') as f:
            pickle.dump(savedata, f)

        # determine if constraints were exceeded, and plot, per target location
        for i in range(len(variation_range_per_parameter[0])):
            within_constraints = []
            outside_constraints = []
            for j in range(num_parameters):
                within_constraints_parameter = []
                outside_constraints_parameter = []
                for k in range(num_simulations):
                    within_gload = constraints[i][j][k][0] < max_g_constraint
                    within_heatflux = constraints[i][j][k][1] < max_heatflux_constraint
                    within_total_heatload = constraints[i][j][k][2] < max_total_heatload_constraint
                    within_final_velocity = constraints[i][j][k][3] < max_final_velocity_constraint
                    succesfull_completion = constraints[i][j][k][4]

                    if within_gload and within_heatflux and within_total_heatload and within_final_velocity and succesfull_completion:
                        input_value = inputs[i][j][k]
                        objective_value = objectives[i][j][k]
                        constraint_value = constraints[i][j][k]
                        within_constraints_parameter.append([input_value, objective_value, constraint_value])
                    else:
                        input_value = inputs[i][j][k]
                        objective_value = objectives[i][j][k]
                        constraint_value = constraints[i][j][k]
                        outside_constraints_parameter.append([input_value, objective_value, constraint_value])
                within_constraints.append(within_constraints_parameter)
                outside_constraints.append(outside_constraints_parameter)

            for j in range(num_parameters):
                input_within_constraints = [within_constraints[j][k][0] for k in range(len(within_constraints[j]))]
                input_outside_constraints = [outside_constraints[j][k][0] for k in range(len(outside_constraints[j]))]

                objective1_within_constraints = [within_constraints[j][k][1][0] for k in range(len(within_constraints[j]))]
                objective2_within_constraints = [within_constraints[j][k][1][1] for k in range(len(within_constraints[j]))]
                objective3_within_constraints = [within_constraints[j][k][1][2] for k in range(len(within_constraints[j]))]

                objective1_outside_constraints = [outside_constraints[j][k][1][0] for k in
                                                  range(len(outside_constraints[j]))]
                objective2_outside_constraints = [outside_constraints[j][k][1][1] for k in
                                                  range(len(outside_constraints[j]))]
                objective3_outside_constraints = [outside_constraints[j][k][1][2] for k in
                                                  range(len(outside_constraints[j]))]

                gload_plots_within_constraints = [within_constraints[j][k][2][0] for k in range(len(within_constraints[j]))]
                heatflux_plots_within_constraints = [within_constraints[j][k][2][1] for k in
                                                     range(len(within_constraints[j]))]
                heatload_plots_within_constraints = [within_constraints[j][k][2][2] for k in
                                                     range(len(within_constraints[j]))]
                final_velocity_plots_within_constraints = [within_constraints[j][k][2][3] for k in
                                                           range(len(within_constraints[j]))]

                gload_plots_outside_constraints = [outside_constraints[j][k][2][0] for k in
                                                   range(len(outside_constraints[j]))]
                heatflux_plots_outside_constraints = [outside_constraints[j][k][2][1] for k in
                                                      range(len(outside_constraints[j]))]
                heatload_plots_outside_constraints = [outside_constraints[j][k][2][2] for k in
                                                      range(len(outside_constraints[j]))]
                final_velocity_plots_outside_constraints = [outside_constraints[j][k][2][3] for k in
                                                            range(len(outside_constraints[j]))]

                # 3 by 1 figure with 3 subplots
                fig, axs = plt.subplots(1, 3, figsize=(10, 5))
                axs = axs.flatten()
                # objectives
                fig.suptitle(
                    'Parameter ' + parameternames[j] + ' against objectives, for ' + variation_range_per_parameter[0][i])
                axs[0].scatter(input_within_constraints, objective1_within_constraints, color='blue',
                               label='Within constraints')
                axs[0].scatter(input_outside_constraints, objective1_outside_constraints, color='red',
                               label='Outside constraints')
                axs[0].set_title('Final distance to target')
                axs[0].set_xlabel('Parameter ' + parameternames_axis[j])
                axs[0].set_ylabel('Final distance to target [m]')
                axs[0].legend()
                axs[0].grid()
                axs[1].scatter(input_within_constraints, objective2_within_constraints, color='blue',
                               label='Within constraints')
                axs[1].scatter(input_outside_constraints, objective2_outside_constraints, color='red',
                               label='Outside constraints')
                axs[1].set_title('Propellant mass')
                axs[1].set_xlabel('Parameter ' + parameternames_axis[j])
                axs[1].set_ylabel('Propellant mass [kg]')
                axs[1].legend()
                axs[1].grid()
                axs[2].scatter(input_within_constraints, objective3_within_constraints, color='blue',
                               label='Within constraints')
                axs[2].scatter(input_outside_constraints, objective3_outside_constraints, color='red',
                               label='Outside constraints')
                axs[2].set_title('Number of bank reversals')
                axs[2].set_xlabel('Parameter ' + parameternames_axis[j])
                axs[2].set_ylabel('number of bank reversals [-]')
                axs[2].legend()
                axs[2].grid()
                plt.tight_layout()
                figname = 'Parameter ' + parameternames[j] + ' MC single objectives ' + variation_range_per_parameter[0][
                    i] + '.png'
                fig.savefig(os.path.join(output_folder, figname))
                plt.show()

                # constraints
                fig, axs = plt.subplots(1, 4, figsize=(10, 5))
                fig.suptitle(
                    'Parameter ' + parameternames[j] + ' against constraints, for ' + variation_range_per_parameter[0][i])
                axs[0].scatter(input_within_constraints, gload_plots_within_constraints, color='blue',
                               label='Within constraints')
                axs[0].scatter(input_outside_constraints, gload_plots_outside_constraints, color='red',
                               label='Outside constraints')
                axs[0].set_title('Max G-load')
                axs[0].set_xlabel('Parameter ' + parameternames_axis[j])
                axs[0].set_ylabel('Max G-load [-]')
                axs[0].grid()
                axs[1].scatter(input_within_constraints, heatflux_plots_within_constraints, color='blue',
                               label='Within constraints')
                axs[1].scatter(input_outside_constraints, heatflux_plots_outside_constraints, color='red',
                               label='Outside constraints')
                axs[1].set_title('Max heat flux')
                axs[1].set_xlabel('Parameter ' + parameternames_axis[j])
                axs[1].set_ylabel('Max heat flux [W/m^2]')
                axs[1].grid()
                axs[2].scatter(input_within_constraints, heatload_plots_within_constraints, color='blue',
                               label='Within constraints')
                axs[2].scatter(input_outside_constraints, heatload_plots_outside_constraints, color='red',
                               label='Outside constraints')
                axs[2].set_title('Max heat load')
                axs[2].set_xlabel('Parameter ' + parameternames_axis[j])
                axs[2].set_ylabel('Max heat load [J/m^2]')
                axs[2].grid()
                axs[3].scatter(input_within_constraints, final_velocity_plots_within_constraints, color='blue',
                               label='Within constraints')
                axs[3].scatter(input_outside_constraints, final_velocity_plots_outside_constraints, color='red',
                               label='Outside constraints')
                axs[3].set_title('Final velocity')
                axs[3].set_xlabel('Parameter ' + parameternames_axis[j])
                axs[3].set_ylabel('Final velocity [m/s]')
                axs[3].grid()
                plt.tight_layout()
                figname = 'Parameter ' + parameternames[j] + ' MC single constraints ' + variation_range_per_parameter[0][
                    i] + '.png'
                fig.savefig(os.path.join(output_folder, figname))
                plt.show()

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

    with multiprocessing.Pool(processes=num_cores) as pool:

        results = pool.map(run_simulation, location_parameters)

    print("\nAll simulations complete!")