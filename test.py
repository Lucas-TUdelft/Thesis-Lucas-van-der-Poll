

import numpy as np


# Load spice kernels
spice_interface.load_standard_kernels()

# Set simulation start epoch
simulation_start_epoch = 0.0  # s
# Set termination conditions
maximum_duration = constants.JULIAN_DAY  # s
termination_altitude = 30.0E3  # m

# Define settings for celestial bodies
bodies_to_create = ['Earth']
# Define Ground station settings (Paris)
station_altitude = 35.0 # m
station_latitude = 48.8575 # deg
station_longitude = 2.3514 # deg
# Define coordinate system
global_frame_origin = 'Earth'
global_frame_orientation = 'J2000'

# Create body settings
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create,
    global_frame_origin,
    global_frame_orientation)

# Create bodies
bodies = environment_setup.create_system_of_bodies(body_settings)

# Create Ground Station
ground_station_settings = environment_setup.ground_station.basic_station(
    "LandingPad",
    [station_altitude, station_latitude, station_longitude],
    element_conversion.geodetic_position_type)
environment_setup.add_ground_station(bodies.get_body("Earth"), ground_station_settings)

# Create capsule
bodies.create_empty_body('Capsule')
constant_angles = np.zeros([3,1])
constant_angles[ 0 ] = 0.2548030601
new_capsule_mass = 9500
bodies.get_body('Capsule').set_constant_mass(new_capsule_mass)
drag_coefficient = 1.5
lift_coefficient = 0.525
aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
    reference_area,
    constant_force_coefficient=[drag_coefficient, 0, lift_coefficient],
    force_coefficients_frame=environment.negative_aerodynamic_frame_coefficients,
)
environment_setup.add_aerodynamic_coefficient_interface(bodies, 'Capsule', new_aerodynamic_coefficient_interface)

# Guidance
environment_setup.add_flight_conditions(bodies, 'Capsule', 'Earth')

aerodynamic_guidance_object = Util.PREDGUID(bodies)
rotation_model_settings = environment_setup.rotation_model.aerodynamic_angle_based(
    'Earth', '', 'BodyFixed', aerodynamic_guidance_object.getAerodynamicAngles )
environment_setup.add_rotation_model( bodies, 'Capsule', rotation_model_settings )

# Create single PropagationTerminationSettings objects
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
termination_settings = propagation_setup.propagator.hybrid_termination(termination_settings_list,
                                                                              fulfill_single_condition=True)

dependent_variables_to_save = [propagation_setup.dependent_variable.mach_number('Capsule', 'Earth'),
                               propagation_setup.dependent_variable.altitude('Capsule', 'Earth'),
                               propagation_setup.dependent_variable.local_aerodynamic_g_load('Capsule', 'Earth'),
                               propagation_setup.dependent_variable.keplerian_state('Capsule', 'Earth'),
                               propagation_setup.dependent_variable.relative_position('Capsule','Earth'),
                               propagation_setup.dependent_variable.relative_velocity('Capsule','Earth'),
                               propagation_setup.dependent_variable.geodetic_latitude('Capsule','Earth'),
                               propagation_setup.dependent_variable.longitude('Capsule','Earth'),
                               propagation_setup.dependent_variable.bank_angle('Capsule','Earth')]

# Define bodies that are propagated and their central bodies of propagation
bodies_to_propagate = ['Capsule']
central_bodies = ['Earth']

# Define accelerations acting on capsule
acceleration_settings_on_vehicle = {
    'Earth': [propagation_setup.acceleration.point_mass_gravity(),
                propagation_setup.acceleration.aerodynamic()]
}

# Create acceleration models.
acceleration_settings = {'Capsule': acceleration_settings_on_vehicle}
acceleration_models = propagation_setup.create_acceleration_models(
    bodies,
    acceleration_settings,
    bodies_to_propagate,
    central_bodies)