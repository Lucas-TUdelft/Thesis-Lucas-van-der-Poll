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
from tudatpy.numerical_simulation import environment_setup
from tudatpy.numerical_simulation import propagation_setup
from tudatpy.numerical_simulation import environment
from tudatpy import numerical_simulation
from tudatpy.astro import element_conversion
#from tudatpy.astro import reference_frames
from tudatpy.kernel.math import interpolators
from tudatpy.util import result2array

import pygmo as pg

# Problem-specific imports
import EntryUtilities_multiprocessing as Util

spice.load_standard_kernels()

###########################################################################
# DEFINE PROBLEM ##########################################################
###########################################################################

class ReentryProblem:
    def __init__(self,
                 simulation_start_epoch,
                 termination_settings,
                 bodies,
                 bounds,
                 dependent_variables_to_save,
                 target_location):

        self.simulation_start_epoch = simulation_start_epoch
        self.termination_settings = termination_settings
        self.bodies = bodies
        self.bounds = bounds
        self.constraints = [10.0, 1.0 * 10 ** 6, 200e6, 600, 5000]
        self.dependent_variables_to_save = dependent_variables_to_save
        self.target_location = target_location

    def get_bounds(self):
        return self.bounds

    def get_number_of_parameters(self):
        return len(self.bounds)

    def get_nobj(self):
        return 2

    def fitness(self, inputs):

        speed = inputs[0]
        heading_angle = inputs[1]
        flight_path_angle = inputs[2]
        guidance_K = inputs[3]
        deadband_c0 = inputs[4]
        deadband_c1 = inputs[5]

        deadband_values = [deadband_c0, deadband_c1]

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

        propagator_settings = propagation_setup.propagator.translational(central_bodies,
                                                                         acceleration_models,
                                                                         bodies_to_propagate,
                                                                         initial_cartesian_state_inertial,
                                                                         self.simulation_start_epoch,
                                                                         None,
                                                                         self.termination_settings,
                                                                         propagation_setup.propagator.cowell,
                                                                         output_variables=dependent_variables_to_save)

        step_size = 1.0
        propagator_settings.integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(
            step_size,
            propagation_setup.integrator.CoefficientSets.rkf_56)

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
        initial_velocity_correction = (initial_cartesian_state_inertial_disconnect -
                                       initial_cartesian_state_inertial)[3:6]
        delta_V = np.linalg.norm(initial_velocity_correction)
        Isp = 360
        g0 = 9.807
        m0 = new_capsule_mass * np.exp(delta_V / (Isp * g0))
        mp = m0 - new_capsule_mass

        n_bank_reversals = aerodynamic_guidance_object.number_of_bank_reversals

        fitness = np.array([mp / 10000, n_bank_reversals + 1])

        if not succesfull_completion:
            fitness = np.array([1e10, 1e10])
        if max_gload >= self.constraints[0]:
            fitness *= (max_gload/self.constraints[0]) * 100
        if max_heatflux >= self.constraints[1]:
            fitness *= (max_heatflux/self.constraints[1]) * 100
        if total_heatload >= self.constraints[2]:
            fitness *= (total_heatload/self.constraints[2]) * 100
        if final_velocity >= self.constraints[3]:
            fitness *= (final_velocity/self.constraints[3]) * 100
        if final_distance_to_target >= self.constraints[4]:
            fitness *= (final_distance_to_target/self.constraints[4])

        return fitness


class optimization:
    def __init__(self, bounds, target_location, optimizer_name):

        # Set simulation start epoch
        self.simulation_start_epoch = 0.0  # s

        # Set termination conditions
        maximum_duration = constants.JULIAN_DAY  # s
        termination_altitude = 30.0E3  # m

        # Define settings for celestial bodies
        bodies_to_create = ['Earth', 'Moon', 'Sun']

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
            'Earth', self.simulation_start_epoch, spice.get_body_gravitational_parameter('Sun'),
            frame_orientation='J2000')
        body_settings.get('Moon').ephemeris_settings = environment_setup.ephemeris.keplerian_from_spice(
            'Moon', self.simulation_start_epoch, spice.get_body_gravitational_parameter('Earth'),
            frame_orientation='J2000')
        body_settings.get('Sun').ephemeris_settings = environment_setup.ephemeris.keplerian_from_spice(
            'Sun', self.simulation_start_epoch, spice.get_body_gravitational_parameter('Sun'),
            frame_orientation='J2000')

        # rotation model
        body_settings.get('Earth').rotation_model_settings = environment_setup.rotation_model.gcrs_to_itrs(
            base_frame='J2000')
        body_settings.get('Earth').gravity_field_settings.associated_reference_frame = 'ITRS'

        # create bodies
        bodies = environment_setup.create_system_of_bodies(body_settings)

        # termination settings
        # Time
        time_termination_settings = propagation_setup.propagator.time_termination(
            self.simulation_start_epoch + maximum_duration,
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
        self.termination_settings = propagation_setup.propagator.hybrid_termination(termination_settings_list,
                                                                                      fulfill_single_condition=True)

        # create ground station
        if target_location == 'Paris':
            station_altitude = 35.0  # m
            station_latitude = np.deg2rad(48.8575)  # rad
            station_longitude = np.deg2rad(2.3514)  # rad
        elif target_location == 'Cabo Verde':
            station_altitude = 37.0  # m
            station_latitude = np.deg2rad(14.9198)  # rad
            station_longitude = np.deg2rad(-23.5073)  # rad
        elif target_location == 'Natal':
            station_altitude = 30.0  # m
            station_latitude = np.deg2rad(-5.7842)  # rad
            station_longitude = np.deg2rad(-35.2000)  # rad
        elif target_location == 'Canarias':
            station_altitude = 0.0  # m
            station_latitude = np.deg2rad(28.2916)  # rad
            station_longitude = np.deg2rad(-16.6291)  # rad
        elif target_location == 'Azores':
            station_altitude = 0.0  # m
            station_latitude = np.deg2rad(37.7412)  # rad
            station_longitude = np.deg2rad(-25.6756)  # rad

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

        self.bodies = bodies

        # dependent variables
        self.dependent_variables_to_save = [propagation_setup.dependent_variable.mach_number('Capsule', 'Earth'),
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


        self.bounds = bounds
        self.target_location = target_location

        if optimizer_name == 'ihs':
            self.optimizer = pg.ihs
        elif optimizer_name == 'nsga2':
            self.optimizer = pg.nsga2
        elif optimizer_name == 'moead':
            self.optimizer = pg.moead
        elif optimizer_name == 'moead_gen':
            self.optimizer = pg.moead_gen
        elif optimizer_name == 'maco':
            self.optimizer = pg.maco
        elif optimizer_name == 'nspso':
            self.optimizer = pg.nspso
        else:
            raise ValueError('Optimizer not recognized, invalid input name')

    def optimize(self, numpops: int, numgens:int, numrepeats: int, seeds: list[float]):
        self.results = []
        self.results_per_generation = []

        integrator = propagation_setup.integrator.runge_kutta_fixed_step_size(
            1.0,
            propagation_setup.integrator.CoefficientSets.rkf_56)
        termination = self.termination_settings
        bodies = self.bodies

        problem_definition = ReentryProblem(self.simulation_start_epoch,
                                            termination,
                                            bodies,
                                            self.bounds,
                                            self.dependent_variables_to_save,
                                            self.target_location)

        problem = pg.problem(problem_definition)

        for i in range(numrepeats):

            seed = seeds[i]

            if self.optimizer == pg.maco:
                algo = pg.algorithm(self.optimizer(seed=seed, ker = numpops - 20))
                algo.set_verbosity(1)
            else:
                algo = pg.algorithm(self.optimizer(seed=seed))
            pop = pg.population(problem, numpops, seed=seed)
            results_this_repeat = []

            for j in range(numgens):
                pop = algo.evolve(pop)
                results_this_repeat.append(results_this_repeat)
            self.results_per_generation.append(results_this_repeat)

            self.results.append(results_this_repeat)
        return None
