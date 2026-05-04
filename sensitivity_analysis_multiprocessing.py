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
import tudatpy.dynamics.simulator as numerical_simulation
from tudatpy.astro import element_conversion
#from tudatpy.astro import reference_frames
from tudatpy.kernel.math import interpolators
from tudatpy.util import result2array

import pygmo as pg

# Problem-specific imports
import EntryUtilities_multiprocessing as Util
import OptimizationUtilities2 as OptUtil

def init_worker():
    spice.load_standard_kernels()

def run_sensitivity_analysis(location):

    try:
        print(f"--- Starting Sensitivity Analysis with Parameter: {location} ---")

        # Set simulation start epoch
        simulation_start_epoch = 0.0  # s
        # Set termination conditions
        maximum_duration = constants.JULIAN_DAY  # s
        termination_altitude = 30.0E3  # m

        # Define settings for celestial bodies
        bodies_to_create = ['Earth', 'Moon', 'Sun']

        # Define Ground station settings
        target_location = location
        if target_location == 'Paris':
            speed = 7505
            heading_angle = np.deg2rad(35.0)
            station_altitude = 35.0  # m
            station_latitude = np.deg2rad(48.8575)  # rad
            station_longitude = np.deg2rad(2.3514)  # rad
            estimated_flight_time = 1080
        elif target_location == 'Cabo Verde':
            station_altitude = 37.0  # m
            station_latitude = np.deg2rad(14.9198)  # rad
            station_longitude = np.deg2rad(-23.5073)  # rad
            estimated_flight_time = 560  # s
            speed = 6.92636753e+03
            heading_angle = 1.19643649e+00
            flight_path_angle = -1.08428164e-02
            guidance_K = 2.10769827e+00
            deadband_c0 = 8.44870329e-02
            deadband_c1 = 6.34367603e-10
        elif target_location == 'Natal':
            station_altitude = 30.0  # m
            station_latitude = np.deg2rad(-5.7842)  # rad
            station_longitude = np.deg2rad(-35.2000)  # rad
            estimated_flight_time = 410  # s
            speed = 6.40229918e+03
            heading_angle = 2.19772681e+00
            flight_path_angle = -1.00050654e-02
            guidance_K = 4.98778632e+00
            deadband_c0 = 2.34701833e-02
            deadband_c1 = 5.08352784e-10
        elif target_location == 'Canarias':
            station_altitude = 0.0  # m
            station_latitude = np.deg2rad(28.2916)  # rad
            station_longitude = np.deg2rad(-16.6291)  # rad
            estimated_flight_time = 740  # s
            speed = 7.23212043e+03
            heading_angle = 8.85707478e-01
            flight_path_angle = -1.00874371e-02
            guidance_K = 4.65507844e+00
            deadband_c0 = 1.52517932e-02
            deadband_c1 = 1.61533571e-09
        elif target_location == 'Azores':
            station_altitude = 0.0  # m
            station_latitude = np.deg2rad(37.7412)  # rad
            station_longitude = np.deg2rad(-25.6756)  # rad
            estimated_flight_time = 730  # s
            speed = 7.35835171e+03
            heading_angle = 5.46522952e-01
            flight_path_angle = -1.26315767e-02
            guidance_K = 5.82641647e+00
            deadband_c0 = 4.21659933e-02
            deadband_c1 = 1.92603511e-09

        deadband_values = [deadband_c0, deadband_c1]

        ground_station_list = [station_altitude, station_latitude, station_longitude]

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

        bodies = environment_setup.create_system_of_bodies(body_settings)

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

        termination_settings = propagation_setup.propagator.hybrid_termination(termination_settings_list,
                                                                                        fulfill_single_condition=True)

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
        script_dir = os.path.dirname(os.path.abspath(__file__))
        lookup_tables_path = os.path.join(script_dir, "AerodynamicLookupTables")
        aero_coefficients_files = {0: os.path.join(lookup_tables_path, "CD_table.txt"),
                                   2: os.path.join(lookup_tables_path, "CL_table.txt")}

        aerodynamics = environment_setup.aerodynamic_coefficients

        aero_coefficient_settings = environment_setup.aerodynamic_coefficients.tabulated_force_only_from_files(
            force_coefficient_files=aero_coefficients_files,
            reference_area=reference_area,
            independent_variable_names=[aerodynamics.altitude_dependent, aerodynamics.mach_number_dependent]
        )

        environment_setup.add_aerodynamic_coefficient_interface(bodies, 'Capsule', aero_coefficient_settings)

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
                                       propagation_setup.dependent_variable.density('Capsule', 'Earth')]

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

        propagator_settings = propagation_setup.propagator.translational(central_bodies,
                                                                         acceleration_models,
                                                                         bodies_to_propagate,
                                                                         initial_cartesian_state_inertial,
                                                                         simulation_start_epoch,
                                                                         None,
                                                                         termination_settings,
                                                                         propagation_setup.propagator.cowell,
                                                                         output_variables=dependent_variables_to_save)

        step_size = 1.0
        propagator_settings.integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(
            step_size,
            propagation_setup.integrator.CoefficientSets.rkf_56)

        script_dir = os.path.dirname(os.path.abspath(__file__))

        # bank angle guidance
        aerodynamic_guidance_object = Util.ApolloGuidance.from_file(
            os.path.join(script_dir, target_location + '_apollo_data_vref.npz'), bodies, deadband_values,
            estimated_flight_time, K=guidance_K)
        rotation_model_settings = environment_setup.rotation_model.aerodynamic_angle_based(
            'Earth', '', 'BodyFixed', aerodynamic_guidance_object.getAerodynamicAngles)
        environment_setup.add_rotation_model(bodies, 'Capsule', rotation_model_settings)

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
        generate_reference_trajectory_file(h0, v0, gamma0, t0, bank_initial, s_target, target_margin, max_loads,
                                           target_location)

        # bank angle guidance
        aerodynamic_guidance_object = Util.ApolloGuidance.from_file(
            os.path.join(script_dir, target_location + '_apollo_data_vref.npz'), bodies, deadband_values,
            estimated_flight_time, K=guidance_K)
        bodies.get_body('Capsule').rotation_model.reset_aerodynamic_angle_function(
            aerodynamic_guidance_object.getAerodynamicAngles)

        # simulation
        dynamics_simulator = numerical_simulation.create_dynamics_simulator(
            bodies,
            propagator_settings)

        baseline_state_history = dynamics_simulator.state_history

        uncertainties = [10,8,6,4,2,1,0.5,0.25,0.1]
        average_position_errors = []
        simulations_within_range = []
        range_miss_all = []

        seeds = [42, 22, 96, 35, 11]

        for l in range(len(seeds)):
            print('seed:', seeds[l])
            rng = np.random.default_rng(seed=seeds[l])

            simulations_within_range_seed = []
            range_miss_seed = []

            for j in range(len(uncertainties)):
                print('uncertainty:', uncertainties[j])
                # monte carlo variation
                sigma_r = uncertainties[j]
                sigma_v = uncertainties[j] / 10

                covariance_matrix = np.diag([
                    sigma_r ** 2,
                    sigma_r ** 2,
                    sigma_r ** 2,
                    sigma_v ** 2,
                    sigma_v ** 2,
                    sigma_v ** 2,
                ])

                N_loops = 10

                within_range = 0
                outside_range = 0
                margin_miss_total = 0

                output_folder = 'SimulationOutput'
                output_subfolder = 'sensitivity analysis'
                output_folder = os.path.join(output_folder, output_subfolder)

                filename = location + '.dat'
                filename = os.path.join(output_folder, filename)

                times = []
                errors = []
                perturbations = []
                maximum_errors = []
                for i in range(N_loops):
                    #print('Loop:', i + 1)
                    perturbation = rng.multivariate_normal(
                        mean=np.zeros(6),
                        cov=covariance_matrix
                    )
                    perturbations.append(str(perturbation))

                    perturbed_state = initial_cartesian_state_inertial + perturbation
                    propagator_settings.initial_states = perturbed_state

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
                    dot_product = np.dot(final_vehicle_position_bodyfixed_unit,
                                         final_groudstation_position_bodyfixed_unit)
                    dot_product = np.clip(dot_product, -1.0, 1.0)
                    final_distance_to_target = Earth_radius * np.arccos(dot_product)
                    # print('final distance to target:', final_distance_to_target)
                    if final_distance_to_target <= 5000:
                        within_range += 1
                        print(final_distance_to_target, 'within range')
                    else:
                        outside_range += 1
                        margin_miss = final_distance_to_target - 5000
                        margin_miss_total += margin_miss
                        print(final_distance_to_target, 'outside of range')

                    # calculate delta-V
                    initial_velocity_correction = (initial_cartesian_state_inertial_disconnect -
                                                   initial_cartesian_state_inertial)[3:6]
                    delta_V = np.linalg.norm(initial_velocity_correction)
                    # print(initial_velocity_correction)
                    Isp = 360
                    g0 = 9.807
                    m0 = bodies.get_body('Capsule').mass * np.exp(delta_V / (Isp * g0))
                    mp = m0 - bodies.get_body('Capsule').mass
                    # print('propellant mass:', mp)

                    '''
                    e_r_mag = []
                    for k in range(len(e_r)):
                        error_magnitude = np.sqrt((e_r[k][0] ** 2) + (e_r[k][1] ** 2) + (e_r[k][2] ** 2))
                        e_r_mag.append(error_magnitude)

                    times.append(time)
                    errors.append(e_r_mag)

                    e_max = max(e_r_mag)
                    maximum_errors.append(e_max)
                    '''
                #average_position_error = np.mean(np.asarray(maximum_errors))
                #average_position_errors.append(average_position_error)
                print('number of simulations within range:', within_range)
                simulations_within_range_seed.append(within_range)

                if outside_range > 0:
                    range_miss_average = margin_miss_total / outside_range
                else:
                    range_miss_average = 0
                range_miss_seed.append(range_miss_average)

            range_miss_all.append(range_miss_seed)

            simulations_within_range.append(simulations_within_range_seed)

        output_to_store = [simulations_within_range, range_miss_all]


        file = open(filename, 'wb')
        pickle.dump(output_to_store, file)
        file.close()

        print(f"--- Finished Simulation with Parameter: {location} ---")
        return {"param": location, "status": "Success"}

    except Exception as e:
        print(f"Error in sim {location}: {e}")
        return {"param": location, "status": f"Failed: {e}"}



if __name__ == "__main__":

    # Define your 5 different parameter sets
    location_parameters = ['Cabo Verde', 'Natal', 'Canarias']

    # Determine how many cores to use (match your 5 parameters)
    num_cores = 3

    print(f"Launching {num_cores} simulations in parallel...")

    ctx = multiprocessing.get_context("spawn")

    with ctx.Pool(processes=num_cores, initializer=init_worker) as pool:

        results = pool.map(run_sensitivity_analysis, location_parameters)

    print("\nAll simulations complete!")