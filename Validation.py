###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import os
import numpy as np

import apollo_utils
from plotting_functions import *

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

###########################################################################
# DEFINE GLOBAL SETTINGS ##################################################
###########################################################################

# Load spice kernels
spice_interface.load_standard_kernels()

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

bodies = environment_setup.create_system_of_bodies(body_settings)

# capsule
bodies.create_empty_body('Capsule')
new_capsule_mass = 4976 # kg
bodies.get_body('Capsule').set_constant_mass(new_capsule_mass)
reference_area = 12.02 # m^2
lookup_tables_path = os.path.join(os.getcwd(),"AerodynamicLookupTables")
aero_coefficients_files = {0: os.path.join(lookup_tables_path, "validation_CD_table.txt"),
                           2: os.path.join(lookup_tables_path, "validation_CL_table.txt")}

aero_coefficient_settings = environment_setup.aerodynamic_coefficients.tabulated_force_only_from_files(
    force_coefficient_files=aero_coefficients_files,
    reference_area=reference_area,
    independent_variable_names=[environment.altitude_dependent, environment.mach_number_dependent]
)

environment_setup.add_aerodynamic_coefficient_interface(bodies, 'Capsule', aero_coefficient_settings)

# flight conditions
environment_setup.add_flight_conditions(bodies, 'Capsule', 'Earth')

# validation bank angle profile
aerodynamic_guidance_object = Util.validation_guidance(bodies)
rotation_model_settings = environment_setup.rotation_model.aerodynamic_angle_based(
    'Earth', '', 'BodyFixed', aerodynamic_guidance_object.getAerodynamicAngles )
environment_setup.add_rotation_model( bodies, 'Capsule', rotation_model_settings )

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
                               propagation_setup.dependent_variable.local_aerodynamic_g_load('Capsule', 'Earth'),
                               propagation_setup.dependent_variable.keplerian_state('Capsule', 'Earth'),
                               propagation_setup.dependent_variable.relative_position('Capsule','Earth'),
                               propagation_setup.dependent_variable.relative_velocity('Capsule','Earth'),
                               propagation_setup.dependent_variable.geodetic_latitude('Capsule','Earth'),
                               propagation_setup.dependent_variable.longitude('Capsule','Earth'),
                               propagation_setup.dependent_variable.bank_angle('Capsule','Earth'),
                               propagation_setup.dependent_variable.relative_speed('Capsule','Earth')
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
radial_distance = spice_interface.get_average_radius('Earth') + 123883.27776
latitude = np.deg2rad(-23.652)
longitude = np.deg2rad(174.928)
speed = 11000
flight_path_angle = np.deg2rad(-6.616)
heading_angle = np.deg2rad(95.0)

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

step_size = 2.0
propagator_settings.integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(
    step_size,
    propagation_setup.integrator.CoefficientSets.rkf_56)

# simulation
dynamics_simulator = numerical_simulation.create_dynamics_simulator(
    bodies,
    propagator_settings)

state_history = dynamics_simulator.state_history
dependent_variables = dynamics_simulator.dependent_variable_history

state_history_array = result2array(state_history)
dependent_variables_array = result2array(dependent_variables)

h = dependent_variables_array[:, 2]
bank = np.rad2deg(dependent_variables_array[:, 18])
vel = dependent_variables_array[:, 19]
g = dependent_variables_array[:, 3]
dependent_variables_time = dependent_variables.keys()

altitude_plot(h, dependent_variables_time)
bank_plot(bank, dependent_variables_time)
velocity_plot(vel, dependent_variables_time)
gload_plot(g, dependent_variables_time)

'''
Validation changes to main:
- mass
- reference area
- CL/CD
- initial state
- guidance
'''