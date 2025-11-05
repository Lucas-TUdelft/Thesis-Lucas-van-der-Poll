###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import math

import numpy as np

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
from tudatpy.kernel.math import geometry

###########################################################################
# VEHICLE SHAPE/AERODYNAMICS UTILITIES ####################################
###########################################################################

def get_capsule_coefficient_interface(capsule_shape: tudatpy.kernel.math.geometry.Capsule) \
        -> tudatpy.kernel.numerical_simulation.environment.HypersonicLocalInclinationAnalysis:
    """
    Function that creates an aerodynamic database for a capsule, based on a set of shape parameters.

    The Capsule shape consists of four separate geometrical components: a sphere segment for the nose, a torus segment
    for the shoulder/edge, a conical frustum for the rear body, and a sphere segment for the rear cap (see Dirkx and
    Mooij, 2016). The code used in this function discretizes these surfaces into a structured mesh of quadrilateral
    panels. The parameters number_of_points and number_of_lines define the number of discretization points (for each
    part) in both independent directions (lengthwise and circumferential). The list selectedMethods defines the type of
    aerodynamic analysis method that is used.

    Parameters
    ----------
    capsule_shape : tudatpy.kernel.math.geometry.Capsule
        Object that defines the shape of the vehicle.

    Returns
    -------
    hypersonic_local_inclination_analysis : tudatpy.kernel.environment.HypersonicLocalInclinationAnalysis
        Database created through the local inclination analysis method.
    """

    # Define settings for surface discretization of the capsule
    number_of_lines = [31, 31, 31, 11]
    number_of_points = [31, 31, 31, 11]
    # Set side of the vehicle (DO NOT CHANGE THESE: setting to true will turn parts of the vehicle 'inside out')
    invert_order = [0, 0, 0, 0]

    # Define moment reference point. NOTE: This value is chosen somewhat arbitrarily, and will only impact the
    # results when you consider any aspects of moment coefficients
    moment_reference = np.array([-0.6624, 0.0, 0.1369])

    # Define independent variable values
    independent_variable_data_points = []
    # Mach
    mach_points = environment.get_default_local_inclination_mach_points()
    independent_variable_data_points.append(mach_points)
    # Angle of attack
    angle_of_attack_points = np.linspace(np.deg2rad(-40),np.deg2rad(40),17)
    independent_variable_data_points.append(angle_of_attack_points)
    # Angle of sideslip
    angle_of_sideslip_points = environment.get_default_local_inclination_sideslip_angle_points()
    independent_variable_data_points.append(angle_of_sideslip_points)

    # Define local inclination method to use (index 0=Newtonian flow)
    selected_methods = [[0, 0, 0, 0], [0, 0, 0, 0]]

    # Get the capsule middle radius
    capsule_middle_radius = capsule_shape.middle_radius
    # Calculate reference area
    reference_area = np.pi * capsule_middle_radius ** 2

    '''
    # Create aerodynamic database
    hypersonic_local_inclination_analysis = environment.HypersonicLocalInclinationAnalysis(
        independent_variable_data_points,
        capsule_shape,
        number_of_lines,
        number_of_points,
        invert_order,
        selected_methods,
        reference_area,
        capsule_middle_radius,
        moment_reference)
    #return hypersonic_local_inclination_analysis
    return AerodynamicCoefficientInterfaceSettings(
        hypersonic_local_inclination_analysis,
        are_angles_in_radians=True,
        reference_area=reference_area,
        force_reference_point=moment_reference
    )
    '''
    drag_coefficient = 1.5
    lift_coefficient = 0.525
    aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
        reference_area,
        constant_force_coefficient=[drag_coefficient, 0, lift_coefficient],
        force_coefficients_frame=environment.negative_aerodynamic_frame_coefficients,
    )
    return aero_coefficient_settings


def set_capsule_shape_parameters(shape_parameters: list,
                                 bodies: tudatpy.kernel.numerical_simulation.environment.SystemOfBodies,
                                 capsule_density: float):
    """
    It computes and creates the properties of the capsule (shape, mass, aerodynamic coefficient interface...).

    Parameters
    ----------
    shape_parameters : list of floats
        List of shape parameters to be optimized.
    bodies : tudatpy.kernel.numerical_simulation.environment.SystemOfBodies
        System of bodies present in the simulation.
    capsule_density : float
        Constant density of the vehicle.

    Returns
    -------
    none
    """
    # Compute shape constraint
    length_limit = shape_parameters[1] - shape_parameters[4] * (1 - np.cos(shape_parameters[3]))
    length_limit /= np.tan(- shape_parameters[3])
    # Add safety factor
    length_limit -= 0.01
    # Apply constraint
    if shape_parameters[2] >= length_limit:
        shape_parameters[2] = length_limit

    # Create capsule
    new_capsule = geometry.Capsule(*shape_parameters[0:5])
    # Compute new body mass
    new_capsule_mass = capsule_density * new_capsule.volume
    # Set capsule mass
    bodies.get_body('Capsule').set_constant_mass(new_capsule_mass)

    # Create aerodynamic interface from shape parameters (this calls the local inclination analysis)
    new_aerodynamic_coefficient_interface = get_capsule_coefficient_interface(new_capsule)
    # Update the Capsule's aerodynamic coefficient interface
    #bodies.get_body('Capsule').aerodynamic_coefficient_interface = new_aerodynamic_coefficient_interface
    environment_setup.add_aerodynamic_coefficient_interface(bodies, 'Capsule', new_aerodynamic_coefficient_interface)


def add_capsule_to_body_system(bodies: tudatpy.kernel.numerical_simulation.environment.SystemOfBodies,
                               shape_parameters: list,
                               capsule_density: float):
    """
    It creates the capsule body object and adds it to the body system, setting its shape based on the shape parameters
    provided.

    Parameters
    ----------
    bodies : tudatpy.kernel.numerical_simulation.environment.SystemOfBodies
        System of bodies present in the simulation.
    shape_parameters : list of floats
        List of shape parameters to be optimized.
    capsule_density : float
        Constant density of the vehicle.

    Returns
    -------
    none
    """
    # Create new vehicle object and add it to the existing system of bodies
    bodies.create_empty_body('Capsule')
    #body_settings.add_empty_settings('Capsule')
    constant_angles = np.zeros([3,1])
    constant_angles[ 0 ] = shape_parameters[ 5 ]
    angle_function = lambda time : constant_angles
    #environment_setup.add_rotation_model( bodies, 'Capsule',
    #                                      environment_setup.rotation_model.aerodynamic_angle_based(
    #                                          'Earth', 'J2000', 'CapsuleFixed', angle_function ))
    #aerodynamic_guidance_object = PREDGUID(bodies)
    #rotation_model_settings = environment_setup.rotation_model.aerodynamic_angle_based(
    #    'Earth', '', 'Capsule_Fixed', aerodynamic_guidance_object.getAerodynamicAngles)
    #environment_setup.add_rotation_model(bodies, 'Capsule', rotation_model_settings)
    #body_settings.get("Capsule").rotation_model_settings = rotation_model_settings

    # Update the capsule shape parameters
    set_capsule_shape_parameters(shape_parameters,
                                 bodies,
                                 capsule_density)


def add_capsule_settings_to_body_system(body_settings,
                                        shape_parameters: list,
                                        capsule_density: float):
    """
    It creates the capsule body object and adds it to the body system, setting its shape based on the shape parameters
    provided.

    Parameters
    ----------
    body_settings :
    shape_parameters : list of floats
        List of shape parameters to be optimized.
    capsule_density : float
        Constant density of the vehicle.

    Returns
    -------
    none
    """
    # Create new vehicle object and add it to the existing system of bodies
    #bodies.create_empty_body('Capsule')
    body_settings.add_empty_settings('Capsule')
    constant_angles = np.zeros([3,1])
    constant_angles[ 0 ] = shape_parameters[ 5 ]
    angle_function = lambda time : constant_angles
    #environment_setup.add_rotation_model( bodies, 'Capsule',
    #                                      environment_setup.rotation_model.aerodynamic_angle_based(
    #                                          'Earth', 'J2000', 'CapsuleFixed', angle_function ))
    #aerodynamic_guidance_object = PREDGUID(body_settings)
    #rotation_model_settings = environment_setup.rotation_model.aerodynamic_angle_based(
    #    'Earth', '', 'Capsule_Fixed', aerodynamic_guidance_object.getAerodynamicAngles)
    #body_settings.get("Capsule").rotation_model_settings = rotation_model_settings

    # Update the capsule shape parameters
    set_capsule_shape_parameters(shape_parameters,
                                 bodies,
                                 capsule_density)


###########################################################################
# PROPAGATION SETTING UTILITIES ###########################################
###########################################################################

def get_initial_state(simulation_start_epoch: float,
                      bodies: tudatpy.kernel.numerical_simulation.environment.SystemOfBodies) -> np.ndarray:
    """
    Converts the initial state to inertial coordinates.

    The initial state is expressed in Earth-centered spherical coordinates.
    These are first converted into Earth-centered cartesian coordinates,
    then they are finally converted in the global (inertial) coordinate
    system.

    Parameters
    ----------
    simulation_start_epoch : float
        Start of the simulation [s] with t=0 at J2000.
    bodies : tudatpy.kernel.numerical_simulation.environment.SystemOfBodies
        System of bodies present in the simulation.

    Returns
    -------
    initial_state_inertial_coordinates : np.ndarray
        The initial state of the vehicle expressed in inertial coordinates.
    """
    # Set initial spherical elements
    radial_distance = spice_interface.get_average_radius('Earth') + 120.0E3
    latitude = np.deg2rad(0.0)
    longitude = np.deg2rad(68.75)
    speed = 7.63E3
    flight_path_angle = np.deg2rad(-0.8)
    heading_angle = np.deg2rad(34.37)

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

    return initial_cartesian_state_inertial


def get_propagator_settings(shape_parameters,
                            bodies,
                            simulation_start_epoch,
                            termination_settings,
                            dependent_variables_to_save,
                            current_propagator = propagation_setup.propagator.cowell ):
    """
    Creates the propagator settings.

    This function creates the propagator settings for translational motion and mass, for the given simulation settings
    Note that, in this function, the entry of the shape_parameters representing the vehicle attitude (angle of attack)
    is processed to redefine the vehice attitude. The propagator settings that are returned as output of this function
    are not yet usable: they do not contain any integrator settings, which should be set at a later point by the user

    Parameters
    ----------
    shape_parameters : list[ float ]
        List of free parameters for the low-thrust model, which will be used to update the vehicle properties such that
        the new thrust/magnitude direction are used. The meaning of the parameters in this list is stated at the
        start of the *Propagation.py file
    bodies : tudatpy.kernel.numerical_simulation.environment.SystemOfBodies
        System of bodies present in the simulation.
    simulation_start_epoch : float
        Start of the simulation [s] with t=0 at J2000.
    termination_settings : tudatpy.kernel.numerical_simulation.propagation_setup.propagator.PropagationTerminationSettings
        Propagation termination settings object to be used
    dependent_variables_to_save : list[tudatpy.kernel.numerical_simulation.propagation_setup.dependent_variable]
        List of dependent variables to save.
    current_propagator : tudatpy.kernel.numerical_simulation.propagation_setup.propagator.TranslationalPropagatorType
        Type of propagator to be used for translational dynamics

    Returns
    -------
    propagator_settings : tudatpy.kernel.numerical_simulation.propagation_setup.integrator.MultiTypePropagatorSettings
        Propagator settings to be provided to the dynamics simulator.
    """

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

    new_angles = np.array([shape_parameters[5], 0.0, 0.0])
    new_angle_function = lambda time : new_angles
    #bodies.get_body('Capsule').rotation_model.reset_aerodynamic_angle_function( new_angle_function )


    # Retrieve initial state
    initial_state = get_initial_state(simulation_start_epoch,bodies)

    # Create propagation settings for the translational dynamics. NOTE: these are not yet 'valid', as no
    # integrator settings are defined yet
    propagator_settings = propagation_setup.propagator.translational(central_bodies,
                                                                     acceleration_models,
                                                                     bodies_to_propagate,
                                                                     initial_state,
                                                                     simulation_start_epoch,
                                                                     None,
                                                                     termination_settings,
                                                                     current_propagator,
                                                                     output_variables=dependent_variables_to_save)
    return propagator_settings

def get_termination_settings(simulation_start_epoch: float,
                             maximum_duration: float,
                             termination_altitude: float) \
        -> tudatpy.kernel.numerical_simulation.propagation_setup.propagator.PropagationTerminationSettings:
    """
    Get the termination settings for the simulation.

    Termination settings currently include:
    - simulation time (one day)
    - lower altitude boundary (25 km)

    Parameters
    ----------
    simulation_start_epoch : float
        Start of the simulation [s] with t=0 at J2000.
    maximum_duration : float
        Maximum duration of the simulation [s].
    termination_altitude : float
        Minimum altitude [m].

    Returns
    -------
    hybrid_termination_settings : tudatpy.kernel.numerical_simulation.propagation_setup.propagator.PropagationTerminationSettings
        Propagation termination settings object.
    """
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
    hybrid_termination_settings = propagation_setup.propagator.hybrid_termination(termination_settings_list,
                                                                                  fulfill_single_condition=True)
    return hybrid_termination_settings

def get_dependent_variable_save_settings() -> list:
    """
    Retrieves the dependent variables to save.

    Currently, the dependent variables saved include:
    - the Mach number
    - the altitude wrt the Earth
    - local aerodynamic g-load
    - vehicle Keplerian state
    - vehicle relative position w.r.t. Earth
    - vehicle relative velocity w.r.t. Earth

    Parameters
    ----------
    none

    Returns
    -------
    dependent_variables_to_save : list[tudatpy.kernel.numerical_simulation.propagation_setup.dependent_variable]
        List of dependent variables to save.
    """
    dependent_variables_to_save = [propagation_setup.dependent_variable.mach_number('Capsule', 'Earth'),
                                   propagation_setup.dependent_variable.altitude('Capsule', 'Earth'),
                                   propagation_setup.dependent_variable.local_aerodynamic_g_load('Capsule', 'Earth'),
                                   propagation_setup.dependent_variable.keplerian_state('Capsule', 'Earth'),
                                   propagation_setup.dependent_variable.relative_position('Capsule','Earth'),
                                   propagation_setup.dependent_variable.relative_velocity('Capsule','Earth'),
                                   propagation_setup.dependent_variable.geodetic_latitude('Capsule','Earth'),
                                   propagation_setup.dependent_variable.longitude('Capsule','Earth'),
                                   propagation_setup.dependent_variable.bank_angle('Capsule','Earth')]
    return dependent_variables_to_save

###########################################################################
# BENCHMARK UTILITIES #####################################################
###########################################################################

def generate_benchmarks(benchmark_step_size,
                        simulation_start_epoch: float,
                        bodies: tudatpy.kernel.numerical_simulation.environment.SystemOfBodies,
                        benchmark_propagator_settings: tudatpy.kernel.numerical_simulation.propagation_setup.propagator.TranslationalStatePropagatorSettings,
                        are_dependent_variables_present: bool,
                        output_path: str = None):
    """
    Function to generate to accurate benchmarks.

    This function runs two propagations with two different integrator settings that serve as benchmarks for
    the nominal runs. The state and dependent variable history for both benchmarks are returned and, if desired,
    they are also written to files (to the directory ./SimulationOutput/benchmarks/) in the following way:
    * benchmark_1_states.dat, benchmark_2_states.dat
        The numerically propagated states from the two benchmarks.
    * benchmark_1_dependent_variables.dat, benchmark_2_dependent_variables.dat
        The dependent variables from the two benchmarks.

    Parameters
    ----------
    benchmark_step_size : float
        Time step of the benchmark that will be used. Two benchmark simulations will be run, both fixed-step 8th order
         (first benchmark uses benchmark_step_size, second benchmark uses 2.0 * benchmark_step_size)
    bodies : tudatpy.kernel.numerical_simulation.environment.SystemOfBodies,
        System of bodies present in the simulation.
    benchmark_propagator_settings
        Propagator settings object which is used to run the benchmark propagations.
    are_dependent_variables_present : bool
        If there are dependent variables to save.
    output_path : str (default: None)
        If and where to save the benchmark results (if None, results are NOT written).

    Returns
    -------
    return_list : list
        List of state and dependent variable history in this order: state_1, state_2, dependent_1_ dependent_2.
    """

    ### CREATION OF THE TWO BENCHMARKS ###
    # Define benchmarks' step sizes
    first_benchmark_step_size = benchmark_step_size  # s
    second_benchmark_step_size = 2.0 * first_benchmark_step_size

    # Create integrator settings for the first benchmark, using a fixed step size RKDP8(7) integrator
    # (the minimum and maximum step sizes are set equal, while both tolerances are set to inf)
    # benchmark_propagator_settings.integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(
    #    first_benchmark_step_size,
    #    propagation_setup.integrator.CoefficientSets.rkdp_87)
    benchmark_propagator_settings.integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(
        first_benchmark_step_size,
        propagation_setup.integrator.CoefficientSets.rkdp_87)
    benchmark_propagator_settings.print_settings.print_dependent_variable_indices = True

    first_dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        bodies,
        benchmark_propagator_settings )

    # Create integrator settings for the second benchmark in the same way
    benchmark_propagator_settings.integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(
        second_benchmark_step_size,
        propagation_setup.integrator.CoefficientSets.rkdp_87)
    benchmark_propagator_settings.print_settings.print_dependent_variable_indices = False

    second_dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        bodies,
        benchmark_propagator_settings )


    ### WRITE BENCHMARK RESULTS TO FILE ###
    # Retrieve state history
    first_benchmark_states = first_dynamics_simulator.state_history
    second_benchmark_states = second_dynamics_simulator.state_history
    # Write results to files
    if output_path is not None:
        save2txt(first_benchmark_states, 'benchmark_1_states.dat', output_path)
        save2txt(second_benchmark_states, 'benchmark_2_states.dat', output_path)
    # Add items to be returned
    return_list = [first_benchmark_states,
                   second_benchmark_states]

    ### DO THE SAME FOR DEPENDENT VARIABLES ###
    if are_dependent_variables_present:
        # Retrieve dependent variable history
        first_benchmark_dependent_variable = first_dynamics_simulator.dependent_variable_history
        second_benchmark_dependent_variable = second_dynamics_simulator.dependent_variable_history
        # Write results to file
        if output_path is not None:
            save2txt(first_benchmark_dependent_variable, 'benchmark_1_dependent_variables.dat',  output_path)
            save2txt(second_benchmark_dependent_variable,  'benchmark_2_dependent_variables.dat',  output_path)
        # Add items to be returned
        return_list.append(first_benchmark_dependent_variable)
        return_list.append(second_benchmark_dependent_variable)

    return return_list

def compare_benchmarks(first_benchmark: dict,
                       second_benchmark: dict,
                       output_path: str,
                       filename: str) -> dict:
    """
    It compares the results of two benchmark runs.

    It uses an 8th-order Lagrange interpolator to compare the state (or the dependent variable, depending on what is
    given as input) history. The difference is returned in form of a dictionary and, if desired, written to a file named
    filename and placed in the directory output_path.

    Parameters
    ----------
    first_benchmark : dict
        State (or dependent variable history) from the first benchmark.
    second_benchmark : dict
        State (or dependent variable history) from the second benchmark.
    output_path : str
        If and where to save the benchmark results (if None, results are NOT written).
    filename : str
        Name of the output file.

    Returns
    -------
    benchmark_difference : dict
        Interpolated difference between the two benchmarks' state (or dependent variable) history.
    """
    # Create 8th-order Lagrange interpolator for first benchmark
    #interpolator_settings = interpolators.lagrange_interpolation(
    #    8, boundary_interpolation=interpolators.extrapolate_at_boundary)
    interpolator_settings = interpolators.lagrange_interpolation(
        8, boundary_interpolation=interpolators.throw_exception_at_boundary, lagrange_boundary_handling=interpolators.LagrangeInterpolatorBoundaryHandling.lagrange_no_boundary_interpolation)
    benchmark_interpolator = interpolators.create_one_dimensional_vector_interpolator(
        first_benchmark, interpolator_settings)
    # Initialize difference dictionaries
    benchmark_difference = dict()
    # Calculate the difference between the states and dependent variables in an iterative manner
    for second_epoch in second_benchmark.keys():
        try:
            benchmark_difference[second_epoch] = benchmark_interpolator.interpolate(second_epoch) - \
                                                    second_benchmark[second_epoch]
        except:
            skipped = True
        #benchmark_difference[second_epoch] = benchmark_interpolator.interpolate(second_epoch) - \
        #                                        second_benchmark[second_epoch]
    # Write results to files
    if output_path is not None:
        save2txt(benchmark_difference, filename, output_path)
    # Return the interpolator
    return benchmark_difference

###########################################################################
# GUIDANCE UTILITIES ######################################################
###########################################################################

class PREDGUID:

    def __init__(self, bodies: environment.SystemOfBodies):

        # Extract vehicle and Earth
        self.vehicle = bodies.get_body("Capsule")
        self.earth = bodies.get_body("Earth")

        # define vehicle initial state (at 0.0)
        self.pos = get_initial_state(0.0, bodies)[0:3]
        self.velocity_I = get_initial_state(0.0, bodies)[3:6]

        self.vehicle_flight_conditions = None
        self.aerodynamic_angle_calculator = None
        self.aerodynamic_coefficient_interface = None

        # Extract ground station
        self.ground_station = self.earth.get_ground_station("LandingPad")

        self.current_time = float("NaN")

        # Define phases indicator:
        # 1: Pre-entry attitude hold and initial roll

        self.Phase = 1

        # Define condition flags:
        self.use_relative_velocities = False # Use inertial velocities: False, use relative velocites: True
        self.Has_entered = False            # Has vehicle entered the sensible atmosphere
        self.Final_Phase = False            # Has vehicle entered final phase
        self.initial_HUNTEST = False        # Has an initial pass through HUNTEST occurred?
        self.HUNTEST_iteration = False      # Has an iteration of HUNTEST occurred?
        self.high_loft = True               # Use high loft algorithm (or low loft)
        self.PredGuid = False               # Run PredGuid routine
        self.stop_bank_comm = False         # stop giving bank angle commands
        self.Gone_past = False              # target has been overshot
        self.skip_bank_reversal = False     # skip bank reversal in this iteration
        self.first_PredGuid = True          # is this the first time through PredGuid

        # Define Constants:
        self.Equatorial_Earth_Rate = 471.434672064 # m/s
        self.VSAT = 7853.53693704 # m/s
        self.g_scaling = 9.81456 # m/s^2
        self.MAX_LD = 0.35 # -
        self.K_rho_filter_gain = 0.05 # -
        self.Earth_rate = 0.0000729211505 # rad/s
        self.Earth_radius = 6378140 # m
        self.V_MIN = self.VSAT/2 # m/s
        self.ATK = 6366707.0736 # m/rad
        self.v_final = 7620 # m/s
        self.K_initial_roll = 13529862.298 # m/s
        self.KA = 19.62912 # m/s^2
        self.RDOT_PHASE2 = -213.36 # m/s
        self.D2 = 53.34 # m/s^2
        self.ref_lD1 = 0.1 # -
        self.ref_LD2 = 0.2 # -
        self.HS = 8686.8 # m
        self.C18 = 152.4 # m/s
        self.Q7F = 1.829 # m/s^2
        self.C1 = 1.25 # -
        self.VLMIN = 5486.4 # m/s
        self.CHOOK = 0.25 # -
        self.CH1 = 0.75 # -
        self.Q19 = 0.2 # -
        self.Q2 = -7920078 # m
        self.Q3 = 0.21245 * (1852 * 3.281) # m/(m/s)
        self.Q5 = 35062064 # m
        self.Q6 = 0.017453 # rad
        self.TOL = 46300 # m
        self.V_Corr_lim = 304.8 # m/s
        self.D0 = 39.624 # m/s^2
        self.C16 = 0.01 # -
        self.C17 = 0.001 # -
        self.max_D = 53.34 # -
        self.KDMIN = 0.1524 # m/s^2
        self.VQUIT = 304.8 # m/s
        self.final_LD = 0.21 # m
        self.GMAX = 10 * self.g_scaling # m/s^2
        self.KLAT = 0.0125 # -
        self.LATBIAS = 0.4 # -
        self.LDCMIR = self.MAX_LD * np.cos(np.deg2rad(15)) # -
        self.CD_est_init = 1.25 # -

        # Define PredGuid constants
        self.VLD = 8503.92 # m/s
        self.J2 = 0.00108263 # -
        self.Earth_mu = 3.986004418 * 10**14 # m^3/s^2
        self.integrator_time_step = 10.0 # s
        self.atmo_alt = 182880 # m
        self.V_loose_req = 9144 # m/s
        self.range_epsilon_1 = 46300 # m
        self.range_epsilon_2 = 1852 # m
        self.delta_phi_min = 0 # rad

        # Initialise variables
        self.angle_of_attack = np.deg2rad(10.0) # rad
        self.bank_angle = 0.0 # rad
        self.Theta = 0  # rad
        ground_station = bodies.get_body("Earth").get_ground_station("LandingPad")
        self.Target_vector0 = ground_station.station_state.get_cartesian_position(0.0) # m
        self.Target_unit_vector = self.Target_vector0 / np.linalg.norm(self.Target_vector0) # -
        self.target_local_east = np.cross(np.array([0,0,1]), self.Target_unit_vector) # -
        self.LATANG = 0.0 # rad
        self.downrange_distance = 0.0 # m
        self.RDOT = 0.0 # m/s
        self.ref_LD = 0.0 # -
        self.HUNTIND = 0.0 # -
        self.Q7 = 0.0 # m/s^2
        self.ALP = 0.0 # -
        self.VL = 0.0 # m/s
        self.gamma_L1 = 0.0 # rad
        self.gamma_L = 0.0 # rad
        self.VBARS = 0.0 # m/s
        self.cos_gamma_L = 0.0 # -
        self.ballistic_e = 0.0 # -
        self.ASKEP = 0.0 # m
        self.ASPF = 0.0 # m
        self.ASPUP = 0.0 # m
        self.ASDWN = 0.0 # m
        self.ASP = 0.0 # m
        self.RDOT_ref_DCRTL = 0.0 # m/s
        self.D_ref_DCTRL = 0.0 # N
        self.K2ROLL = 0.0 # -
        self.K1ROLL = 0.0 # -
        self.PREDANGL = 0.0 # m
        self.X = 0.0 # m/s
        self.Y = 0.0 # rad
        self.Krho_est = 1.0 # -
        self.LD_comm = 0.0 # -
        self.Dbar = np.asarray([0,0,0]) # m/s^2
        self.aero_force = 0.0 # m/s^2
        self.V1 = 0.0 # m/s

        # Final Phase Reference Table
        # NOTE: these values are all in ft and nm, convert them to metric before use
        self.final_vel_list = [0, 236, 862, 1487, 2113, 2739, 3364, 3990, 4616, 5241, 5867, 6493, 7118, 7744, 8370,
                               8995, 9621, 10246, 10872, 11498, 12123, 12749, 13375, 14000, 14626, 15252, 15877, 16503,
                               17129, 17754, 18380, 19006, 19631, 20257, 20882, 21508, 22134, 22759, 23385, 24011,
                               24636, 25262, 35000]
        self.final_phase_table = {
            '0':  [- 230.68, 34.219, 0.0000754,  -0.03546, 0.003927, 1],
            '236' : [-230.68, 34.219, 0.0000754, -0.03546, 0.003927, 1],
            '862 ' : [-660.22, 42.245, 0.002553, -0.03072, 3.4691, 18.295],
            '1487 ' : [-702.79, 50.763, 0.003253, -0.05057, 7.4404, 24.982],
            '2113' : [-679.37, 60.767, 0.004175, -0.06528, 11.696, 31.059],
            '2739' : [-650.97, 70.409, 0.005342, -0.07676, 16.218, 37.987],
            '3364' : [-627.14, 79.309, 0.006714, -0.08658, 21.014, 46.297],
            '3990' : [-608.56, 86.488, 0.008289, -0.09333, 26.128, 56.361],
            '4616' : [-593.43, 92.557, 0.01008, -0.10427, 31.608, 68.483],
            '5241' : [-581.75, 97.704, 0.012064, -0.11493, 37.458, 82.923],
            '5867' : [-573.14, 101.93, 0.014238, -0.12565, 43.715, 99.974],
            '6493' : [-567.11, 105.21, 0.016599, -0.13665, 50.396, 119.86],
            '7118' : [-563, 107.52, 0.019151, -0.1482, 57.529, 142.8],
            '7744' : [-560.07, 108.85, 0.021914, -0.1606, 65.181, 169.11],
            '8370' : [-557.53, 109.22, 0.024903, -0.17412, 73.399, 199.04],
            '8995' : [-554.57, 108.65, 0.028139, -0.18906, 82.237, 232.84],
            '9621' : [-550.32, 107.19, 0.031664, -0.20582, 91.811, 270.97],
            '10246' : [-543.93, 104.94, 0.035505, -0.22474, 102.18, 156.84],
            '10872' : [-534.58, 101.99, 0.039719, -0.24629, 113.49, 180.76],
            '11498' : [-521.47, 98.491, 0.044358, -0.27089, 125.86, 207.48],
            '12123' : [-504.59, 95.413, 0.04944, -0.31708, 139.34, 237.23],
            '12749' : [-484.8, 92.244, 0.05497, -0.34261, 154, 270.36],
            '13375' : [-461.93, 88.864, 0.061012, -0.37169, 169.96, 307.15],
            '14000' : [-436.05, 85.421, 0.067631, -0.4206, 187.29, 347.83],
            '14626' : [-408.27, 82.869, 0.074858, -0.46954, 206.08, 392.94],
            '15252' : [-380.15, 80.383, 0.082694, -0.50208, 226.28, 442.78],
            '15877' : [-352.5, 78.015, 0.091211, -0.53858, 247.92, 497.64],
            '16503' : [-326.17, 75.797, 0.10053, -0.57955, 271.12, 558.18],
            '17129' : [-302.25, 73.742, 0.11075, -0.62545, 295.88, 624.83],
            '17754' : [-281.91, 71.832, 0.12197, -0.67694, 322.21, 698.07],
            '18380' : [-266.14, 69.999, 0.13437, -0.73528, 350.22, 778.82],
            '19006' : [-255.98, 68.151, 0.14812, -0.80222, 379.96, 867.78],
            '19631' : [-252.19, 66.157, 0.16342, -0.88035, 411.52, 965.7],
            '20257' : [-255.29, 63.845, 0.18066, -0.9742, 445.22, 1074],
            '20882' : [-265.54, 61.031, 0.20026, -1.0903, 481.35, 1193.7],
            '21508' : [-282.96, 57.488, 0.22299, -1.2403, 520.65, 1327],
            '22134' : [-307.2, 52.985, 0.24995, -1.4438, 564.09, 1476],
            '22759' : [-337.54, 47.281, 0.28297, -1.738, 613.32, 1643.8],
            '23385' : [-372.79, 40.091, 0.32568, -2.2064, 671.64, 1836],
            '24011' : [-410.49, 31.14, 0.3856, -3.069, 745.54, 2061],
            '24636' : [-444.44, 20.128, 0.48396, -5.1679, 852.47, 2336.2],
            '25262' : [-440.95, 6.3697, 0.74897, -18.237, 1090.1, 2722.7],
            '35000' : [-440.95, 6.3697, 0.74897, -18.237, 1090.1, 2722.7]
        }

    def getAerodynamicAngles(self, current_time: float):

        print(f"[DEBUG] getAerodynamicAngles called at t = {current_time}")
        self.updateGuidance( current_time )
        print(f"[DEBUG] AoA = {self.angle_of_attack}, Bank = {self.bank_angle}")
        return np.array([self.angle_of_attack, 0.0, self.bank_angle])

    # PredGuid sequence
    def PredGuid(self):
        ''' PredGuid Subroutine '''

        # calculate range that must be covered during PredGuid phase (from current time to start of final phase
        # calculate final phase range
        VL_fps = self.VL / 3.281
        if VL_fps <= 25262:
            searching_interpolation = True
            # use interpolation
            for i in range(len(self.final_vel_list)):
                if VL_fps <= self.final_vel_list[i]:
                    if searching_interpolation:
                        V_upper = self.final_vel_list[i]
                        V_lower = self.final_vel_list[i - 1]
                        searching_interpolation = False

            upper_list = final_phase_table[str(V_upper)]
            lower_list = final_phase_table[str(V_lower)]

            RDOTREF = np.interp(VL_fps, [V_lower, V_upper], [lower_list[0], upper_list[0]])
            DREFR = np.interp(VL_fps, [V_lower, V_upper], [lower_list[1], upper_list[1]])
            F2 = np.interp(VL_fps, [V_lower, V_upper], [lower_list[2], upper_list[2]])
            F1 = np.interp(VL_fps, [V_lower, V_upper], [lower_list[3], upper_list[3]])
            RTOGO = np.interp(VL_fps, [V_lower, V_upper], [lower_list[4], upper_list[4]])
            F3 = np.interp(VL_fps, [V_lower, V_upper], [lower_list[5], upper_list[5]])

            ASPF_nm = RTOGO + F2 * (-1 * self.gamma_L * VL_fps - RDOTREF) + F1 * \
                      ((self.Q7 / 3.281) - DREFR)
            self.ASPF = ASPF_nm * 1852

        else:
            # use linear about that point
            self.ASPF = self.Q2 + self.Q3 * self.VL + self.Q5 * (self.Q6 - self.gamma_L)

        # calculate final phase range
        self.PredGuid_target = self.downrange_distance - self.ASPF

        # decide whether to run pc_sequencer: run if range long enough or pc_sequencer has not been run before
        if self.PredGuid_target >= 185200 or self.first_PredGuid == True:
            # initialise inputs
            CD_est = self.CD_est_init
            LD_est = self.MAX_LD
            PG_Krho_est = self.Krho_est
            CPhi_Desired = self.LD_comm / self.MAX_LD
            Bank_sign = 1
            position = self.vehicle.position
            velocity = self.vehicle.velocity
            acceleration = self.Dbar
            altitude = self.altitude
            velocity_Mag = np.linalg(velocity)
            PG_Q7 = self.Q7 + self.KDMIN
            if self.RDOT <= 0 and self.aero_force >= PG_Q7:
                IND_ini = 0
            else:
                IND_ini = 1

            if self.first_PredGuid == True:
                self.first_PredGuid = False
                CPhi_Desired = 0.0
                Lift_INC_CAPTURE = -10.0
                MAX_nr_runs = 10


            # reinitialise PC variables
            bracket = False
            N_high = 0
            N_low = 0
            N_capt = 0
            N_good = 0
            N_esc = 0
            N_crash = 0
            cos_capt = 99999999
            cos_esc = 99999999
            cos_crash = 99999999
            cos_phi_try = 99999999
            cos_bracket = 99999999
            cos_extrap = 99999999
            r_extrap = 99999999
            r_bracket = 99999999
            delta_r = 0
            F_b = 0
            phi_try_last = 0

            # if velocity is lower than lift down velocity, terminate lift down modeling by resetting model
            # lift down flag to false
            if velocity_Mag <= self.VLD:
                self.Model_Lift_Down = False

            # select apogee criteria ###################################################

            if velocity_Mag >= self.V_loose_req:
                # use looser requirement
                range_req = self.range_epsilon_1
            else:
                range_req = self.range_epsilon_2

            # calculate desired bank angle by executing steps a number of times
            nr_run = 1
            while nr_run <= MAX_nr_runs:
                # calculate bank angle to try and its cosine using corrector

                # CORRECTOR #######################################
                if nr_run == 1:
                    # use method 1
                    cos_phi_try = CPhi_Desired
                # use bracketed solution methods
                elif bracket == True:
                    # check if number of low solutions is not zero
                    if N_low >= 1 and N_high >= 1:
                        # we have a high and low solution, use method 2
                        # interpolate between a high and low solution
                        cos_phi_try = (cos_bracket[0] + cos_bracket[1]) / 2

                    elif N_low >= 1 and N_esc >= 1:
                        # there is a low solution and an escape, use method 3
                        # interpolate between low and escape solutions
                        cos_phi_try = cos_bracket[0] + ((cos_esc - cos_bracket[0]) * 0.5)

                else:
                    # no bracketed solution
                    # check if there is only 1 good solution
                    if N_good == 1:
                        # no good solutions, use method 4
                        if N_crash == 0:
                            # only escape solutions, march out of capture region by decreasing L/D
                            cos_phi_try = np.cos(np.arccos(cos_capt) + np.deg2rad(Lift_INC_CAPTURE))

                        elif N_esc == 0:
                            # only capture solutions, march out of capture region by increasing L/D
                            cos_phi_try = np.cos(np.arccos(cos_capt) - np.deg2rad(Lift_INC_CAPTURE))


                    elif N_good >= 2:
                        # extrapolate from the 2 low or 2 high solutions
                        cos_phi_try = ((cos_extrap[1] - cos_extrap[0]) / (r_extrap[1] - r_extrap[0])) \
                                      * (self.PredGuid_target - r_extrap[1]) + cos_extrap[1]

                    elif N_good == 0:
                        # no good solutions, use method 4
                        if N_crash == 0:
                            # only escape solutions, march out of capture region by decreasing L/D
                            cos_phi_try = np.cos(np.arccos(cos_capt) + np.deg2rad(Lift_INC_CAPTURE))

                        elif N_esc == 0:
                            # only capture solutions, march out of capture region by increasing L/D
                            cos_phi_try = np.cos(np.arccos(cos_capt) - np.deg2rad(Lift_INC_CAPTURE))

                # ensure cosine of bank angle is within minimum and maximum values
                cos_phi_try = np.median([np.cos(0), cos_phi_try, np.cos(np.pi)])

                # calculate bank angle to try
                phi_try = np.arccos(cos_phi_try)

                #################################################################

                # calculate predicted range using predictor

                # PREDICTOR #####################################################

                # initialise variables
                r_pred = position
                v_pred = velocity
                r_pred_mag2 = np.dot(position, position)
                r_pred_mag = np.sqrt(r_pred_mag2)
                v_pred_mag2 = np.dot(velocity, velocity)
                v_pred_mag = np.sqrt(v_pred_mag2)

                cos_pred = np.cos(phi_try)
                sin_pred = np.sin(phi_try)

                LD_pred = LD_est

                pred_capt = False

                sin_LD_sign = Bank_sign * np.sin(np.deg2rad(115))

                BC = self.vehicle.mass / (CD_est * self.aerodynamic_coefficient_interface.reference_area)

                # execute predictor loop a number of times
                predicting = True
                while predicting:
                    # execute the 4th order Runge-Kutta Integration Loop 4 times
                    rk4_nr = 1
                    while rk4_nr <= 4:
                        # calculate relative velocity
                        V_rel = v_pred - self.Earth_rate * np.cross(np.asarray([0, 0, 1], r_pred))
                        V_rel_mag2 = np.dot(V_rel, V_rel)
                        V_rel_mag = np.sqrt(V_rel_mag2)

                        # calculate density
                        H_ref = 73921.60
                        rho_ref = 0.0000512644
                        H_pred = r_pred_mag - self.Earth_radius
                        H_norm = H_pred / H_ref
                        if H_pred >= 121920:
                            HS_norm = 60960 / H_ref
                        else:
                            HS_norm = (0.3168398 * H_norm ** 4) + (-1.59503 * H_norm ** 3) + (3.023997 * H_norm ** 2) + \
                                      (- 2.574749 * H_norm) + 0.9154583
                        rho_pred = rho_ref * np.exp((1 - H_norm) / HS_norm)

                        # compute aerodynamic accelerations
                        a_d = PG_Krho_est * rho_pred * V_rel_mag2 / (BC ** 2)
                        a_l = LD_pred * a_d

                        # compute lift vector
                        vel_unit = V_rel / V_rel_mag
                        I_lat_c = np.cross(vel_unit, r_pred)
                        I_lat_c_mag_2 = np.dot(I_lat_c, I_lat_c)
                        I_lat_c_mag = np.sqrt(I_lat_c_mag_2)
                        I_lat = I_lat_c / I_lat_c_mag

                        if self.Model_Lift_Down and v_pred <= self.VLD:
                            I_lift = np.cross(I_lat, vel_unit) * np.cos(np.deg2rad(115)) + \
                                     I_lat * sin_LD_sign
                        else:
                            I_lift = np.cross(I_lat, vel_unit) * cos_pred + I_lat * sin_pred

                        a_acc = a_l * I_lift - a_d * vel_unit

                        # compute gravity acceleration with J2
                        U_pred = r_pred / r_pred_mag
                        Z_pred = np.dot(U_pred, np.asarray([0, 0, 1]))
                        U_pred = U_pred + ((3 / 2) * self.J2) * ((self.Earth_radius / r_pred_mag) ** 2) * \
                                 (((1 - 5 * (Z_pred ** 2)) * U_pred) - (2 * Z_pred * np.asarray([0, 0, 1])))
                        a_g = (-1 * self.Earth_mu / r_pred_mag2) * U_pred

                        # compute total acceleration
                        a_pred = a_acc + a_g

                        # perform integrator
                        if rk4_nr == 1:
                            # set values for original position and velocity
                            r_orig = r_pred
                            v_orig = v_pred

                            # set accumulated velocity and acceleration
                            v_accum = v_pred
                            a_accum = a_pred

                            # perform first integration step
                            r_pred = r_orig + 0.5 * self.integrator_time_step * v_pred
                            v_pred = v_orig + 0.5 * self.integrator_time_step * a_pred

                        if rk4_nr == 2:
                            # set accumulated velocity and acceleration
                            v_accum = v_accum + 2 * v_pred
                            a_accum = a_accum + 2 * a_pred

                            # perform second integration step
                            r_pred = r_orig + 0.5 * self.integrator_time_step * v_pred
                            v_pred = v_orig + 0.5 * self.integrator_time_step * a_pred

                        if rk4_nr == 3:
                            # set accumulated velocity and acceleration
                            v_accum = v_accum + 2 * v_pred
                            a_accum = a_accum + 2 * a_pred

                            # perform third integration step
                            r_pred = r_orig + 0.5 * self.integrator_time_step * v_pred
                            v_pred = v_orig + 0.5 * self.integrator_time_step * a_pred

                        if rk4_nr == 4:
                            # perform fourth integration step
                            r_pred = r_orig + (self.integrator_time_step / 6) * (v_accum + v_pred)
                            v_pred = v_orig + (self.integrator_time_step / 6) * (a_accum + a_pred)

                        # compute state vectors
                        r_pred_mag2 = np.dot(r_pred, r_pred)
                        r_pred_mag = np.sqrt(r_pred_mag2)

                        v_pred_mag2 = np.dot(v_pred, v_pred)
                        v_pred_mag = np.sqrt(v_pred_mag)

                        rk4_nr = rk4_nr + 1

                    # check for atmospheric exit
                    # calculate centrifugal velocity
                    r_dot_pred = np.dot(v_pred, r_pred) / r_pred_mag
                    gamma_pred = np.arcsin(r_dot_pred / v_pred_mag)

                    # check for crash
                    if H_pred <= 0:
                        # crash type solution
                        predicting = False
                        range_pred = 99999999
                        cos_crash = cos_phi_try
                        N_crash = N_crash + 1

                    # check for escape
                    if H_pred >= self.atmo_alt:
                        # escape type solution
                        predicting = False
                        range_pred = -99999999
                        cos_esc = cos_phi_try
                        N_esc = N_esc + 1

                    # check if downcontrol has been ended
                    if v_pred <= self.V1:
                        IND_ini = 1
                    # check for capture
                    if IND_ini == 1 and a_acc >= PG_Q7 and r_dot_pred <= 0:
                        # capture type solution
                        predicting = False
                        pred_capt = True
                        range_pred = np.arccos(np.dot(self.Target_unit_vector, r_pred)) * self.ATK

                ###################################################################

                # compute range miss
                delta_r = range_pred - self.PredGuid_target
                delta_r_norm = abs(delta_r)

                # if range miss is less than the range correct criteria, try was acceptable
                if delta_r_norm <= range_req:
                    # try was acceptable, set desired cosine of bank angle to bank angle tried and exit procedure
                    CPhi_Desired = cos_phi_try
                    nr_run = MAX_nr_runs + 1

                # check if full lift down is needed
                if cos_phi_try <= np.cos(np.pi):
                    # full lift down is needed, exit the procedure
                    CPhi_Desired = np.cos(np.pi)
                    nr_run = MAX_nr_runs + 1

                # check if full lift up is needed
                if cos_phi_try >= np.cos(0):
                    # full lift up is needed, exit the procedure
                    CPhi_Desired = np.cos(0)
                    nr_run = MAX_nr_runs + 1

                # determine nature of the solution
                if pred_capt:
                    # solution captured, increment number of captured solutions
                    N_capt = N_capt + 1
                    # set capture cosine to cosine of bank angle tried
                    cos_capt = cos_phi_try
                else:
                    # save solution as a good solution, increment number of good solutions
                    N_good = N_good + 1
                    # rotate new good solutions into extrapolation variables
                    cos_extrap[1] = cos_extrap[0]
                    cos_extrap[0] = cos_phi_try
                    r_extrap[1] = r_extrap[0]
                    r_extrap[0] = range_pred

                    if delta_r >= 0:
                        # solution is high, increment number of high solutions
                        N_high = N_high + 1
                        # update high bracket terms
                        cos_bracket[0] = cos_phi_try
                        r_bracket[0] = range_pred

                    else:
                        # solution is low, inrement number of low solutions
                        N_low = N_low + 1
                        # update low bracket terms
                        cos_bracket[1] = cos_phi_try
                        r_bracket[1] = range_pred

                # check if there is a high and low solution
                if N_high >= 1 and N_low >= 1:
                    # solution is bracketed
                    bracket = True

                if bracket:
                    # calculate delta bank angle from the last try
                    delta_phi = abs(phi_try - phi_try_last)
                    # check if bank angle change is less than minimum allowable bank angle change
                    if delta_phi <= delta_phi_min:
                        CPhi_Desired = np.cos((phi_try + phi_try_last) / 2)

                # set previous bank angle try to current bank angle
                phi_try_last = phi_try

                # check if maximum number of runs has been reached
                if nr_run == MAX_nr_runs:
                    # perform corrector procedure once more
                    if nr_run == 1:
                        # use method 1
                        cos_phi_try = CPhi_Desired
                    # use bracketed solution methods
                    elif bracket == True:
                        # check if number of low solutions is not zero
                        if N_low >= 1 and N_high >= 1:
                            # we have a high and low solution, use method 2
                            # interpolate between a high and low solution
                            cos_phi_try = (cos_bracket[0] + cos_bracket[1]) / 2

                        elif N_low >= 1 and N_esc >= 1:
                            # there is a low solution and an escape, use method 3
                            # interpolate between low and escape solutions
                            cos_phi_try = cos_bracket[0] + ((cos_esc - cos_bracket[0]) * 0.5)

                    else:
                        # no bracketed solution
                        # check if there is only 1 good solution
                        if N_good == 1:
                            # no good solutions, use method 4
                            if N_crash == 0:
                                # only escape solutions, march out of capture region by decreasing L/D
                                cos_phi_try = np.cos(np.arccos(cos_capt) + np.deg2rad(Lift_INC_CAPTURE))

                            elif N_esc == 0:
                                # only capture solutions, march out of capture region by increasing L/D
                                cos_phi_try = np.cos(np.arccos(cos_capt) - np.deg2rad(Lift_INC_CAPTURE))

                        elif N_good >= 2:
                            # extrapolate from the 2 low or 2 high solutions
                            cos_phi_try = ((cos_extrap[1] - cos_extrap[0]) / (r_extrap[1] - r_extrap[0])) \
                                          * (self.PredGuid_target - r_extrap[1]) + cos_extrap[1]

                        elif N_good == 0:
                            # no good solutions, use method 4
                            if N_crash == 0:
                                # only escape solutions, march out of capture region by decreasing L/D
                                cos_phi_try = np.cos(np.arccos(cos_capt) + np.deg2rad(Lift_INC_CAPTURE))

                            elif N_esc == 0:
                                # only capture solutions, march out of capture region by increasing L/D
                                cos_phi_try = np.cos(np.arccos(cos_capt) - np.deg2rad(Lift_INC_CAPTURE))

                    # ensure cosine of bank angle is within minimum and maximum values
                    cos_phi_try = np.median([np.cos(0), cos_phi_try, np.cos(np.pi)])

                    # calculate bank angle to try
                    phi_try = np.arccos(cos_phi_try)

                    # set cosine of desired bank angle to the calculated bank angle to try
                    CPhi_Desired = cos_phi_try

        # outputs from PredGuid routine
        LD_command = CPhi_Desired * self.MAX_LD
        if delta_r_norm <= (1.852 * 10**6):
            velmag_output = v_pred_mag
            gamma_L_output = - r_dot_pred / velmag_output

        return(LD_command, delta_r, velmag_output, gamma_L_output)


    # call at each simulation step to get bank angle
    def updateGuidance(self, current_time: float):
        print(f"[DEBUG] updateGuidance called at t = {current_time}")
        if(math.isnan( current_time)):
            self.current_time = float("NaN")
        elif( current_time != self.current_time ):
            print(current_time)
            # Get the (constant) angular velocity of the Earth body
            #earth_angular_velocity = np.linalg.norm(self.earth.body_fixed_angular_velocity)
            # Get the distance between the vehicle and the Earth bodies
            #earth_distance = np.linalg.norm(self.vehicle.position)
            # Get the (constant) mass of the vehicle body
            body_mass = self.vehicle.mass

            # Extract flight conditions
            if self.vehicle_flight_conditions == None:
                self.vehicle_flight_conditions = self.vehicle.flight_conditions
                self.aerodynamic_angle_calculator = self.vehicle_flight_conditions.aerodynamic_angle_calculator
                self.aerodynamic_coefficient_interface = self.vehicle_flight_conditions.aerodynamic_coefficient_interface

            # Extract the current Mach number, airspeed, and air density from the flight conditions
            mach_number = self.vehicle_flight_conditions.mach_number
            airspeed = self.vehicle_flight_conditions.airspeed
            density = self.vehicle_flight_conditions.density
            altitude = self.vehicle_flight_conditions.altitude

            # Extract vehicle inertial position and velocity from vehicle state
            #self.pos = self.vehicle_flight_conditions.body_centered_body_fixed_state[0:3]
            self.pos = self.vehicle.position
            self.velocity_I = self.vehicle_flight_conditions.body_centered_body_fixed_state[3:6]

            # Update the variables on which the aerodynamic coefficients are based (AoA and Mach)
            #current_aerodynamics_independent_variables = [self.angle_of_attack, mach_number]

            # Update the aerodynamic coefficients
            #self.aerodynamic_coefficient_interface.update_coefficients(
            #    current_aerodynamics_independent_variables, current_time)

            # Extract the current force coefficients (in order: C_D, C_S, C_L)
            current_force_coefficients = self.aerodynamic_coefficient_interface.current_force_coefficients
            # Extract the (constant) reference area of the vehicle
            aerodynamic_reference_area = self.aerodynamic_coefficient_interface.reference_area

            # Get the heading, flight path, and latitude angles from the aerodynamic angle calculator
            heading = self.aerodynamic_angle_calculator.get_angle(environment.heading_angle)
            flight_path_angle = self.aerodynamic_angle_calculator.get_angle(environment.flight_path_angle)
            latitude = self.aerodynamic_angle_calculator.get_angle(environment.latitude_angle)

            ###################################################################################
            # Targeting
            ###################################################################################

            # if it is the first time step, run targeting sequence twice to start convergence of LATANG to correct sign
            if current_time == 0.0:
                initialising = True
                counter = 0
                while initialising:
                    # 1. Calculate velocity vector to use
                    pos_unit = self.pos / np.linalg.norm(self.pos)
                    V_I_unit = self.velocity_I / np.linalg.norm(self.velocity_I)
                    if not self.use_relative_velocities:
                        velocity = self.velocity_I
                    else:
                        velocity = self.velocity_I - self.Equatorial_Earth_Rate * np.dot(np.array([0, 0, 1]), pos_unit)

                    # 2. calculate parameters
                    V = np.linalg.norm(velocity)
                    VSQ = (V ** 2) / (self.VSAT ** 2)
                    LEQ = (VSQ - 1) * self.g_scaling
                    self.RDOT = np.dot(velocity, pos_unit)
                    UNIbar_vec = np.cross(velocity, pos_unit)
                    #UNIbar = UNIbar_vec / np.linalg.norm(UNIbar_vec)
                    self.Dbar = np.asarray([(current_force_coefficients[0] * 0.5 * density * (airspeed ** 2) * aerodynamic_reference_area),
                                            (current_force_coefficients[1] * 0.5 * density * (airspeed ** 2) * aerodynamic_reference_area),
                                            (current_force_coefficients[2] * 0.5 * density * (airspeed ** 2) * aerodynamic_reference_area)
                                            ])
                    self.aero_force = np.linalg.norm(self.Dbar)

                    # 3. estimate density bias factor
                    rho = (2 * body_mass * self.aero_force) / \
                          ((airspeed ** 2) * aerodynamic_reference_area * current_force_coefficients[0] * np.sqrt(
                              (1 + self.MAX_LD ** 2)))

                    Krho_now = rho / density
                    self.Krho_est = self.K_rho_filter_gain * Krho_now + (1 - self.K_rho_filter_gain) * self.Krho_est

                    # 4. estimate rotation of target
                    if not self.use_relative_velocities:
                        if self.Final_Phase:
                            angle_to_pred_target = self.Earth_rate * (1000 * self.Theta + current_time)
                        else:
                            angle_to_pred_target = self.Earth_rate * (
                                        ((self.Earth_rate * self.Theta) / V) + current_time)
                            if V <= self.V_MIN:
                                self.use_relative_velocities = True
                    else:
                        angle_to_pred_target = self.Earth_rate * current_time

                    # 5. calculate rotated target vector, crossrange angle, and downrange to go:
                    self.target_local_east = np.cross(np.array([0, 0, 1]), self.Target_unit_vector)
                    self.Target_unit_vector = self.Target_vector0 / np.linalg.norm(self.Target_vector0) \
                                              + self.Target_unit_vector * (np.cos(angle_to_pred_target) - 1) \
                                              + self.target_local_east * np.sin(angle_to_pred_target)

                    self.LATANG = np.dot(self.Target_unit_vector, np.cross(V_I_unit, pos_unit))
                    self.Theta = np.arccos(np.dot(self.Target_unit_vector, pos_unit))
                    self.downrange_distance = self.Theta * self.ATK

                    counter = counter+1
                    # check if targeting sequence has been run twice
                    if counter >= 2:
                        initialising = False
                        # initialise K2ROLL
                        self.K2ROLL = -1 * np.sign(self.LATANG)



            # 1. Calculate velocity vector to use
            pos_unit = self.pos / np.linalg.norm(self.pos)
            V_I_unit = self.velocity_I / np.linalg.norm(self.velocity_I)
            if not self.use_relative_velocities:
                velocity = self.velocity_I
            else:
                velocity = self.velocity_I - self.Equatorial_Earth_Rate * np.dot(np.array([0,0,1]), pos_unit)

            # 2. calculate parameters
            V = np.linalg.norm(velocity)
            VSQ = (V**2)/(self.VSAT**2)
            LEQ = (VSQ - 1) * self.g_scaling
            self.RDOT = np.dot(velocity, pos_unit)
            UNIbar_vec = np.cross(velocity, pos_unit)
            #UNIbar = UNIbar_vec / np.linalg.norm(UNIbar_vec)
            self.aero_force = (current_force_coefficients[0] + current_force_coefficients[2]) \
                         * 0.5 * density * (airspeed**2) * aerodynamic_reference_area

            # 3. estimate density bias factor
            rho = (2 * body_mass * self.aero_force)\
                  /((airspeed**2) * aerodynamic_reference_area * self.CD_est_init * np.sqrt((1 + self.MAX_LD**2)))

            Krho_now = rho / density
            self.Krho_est = self.K_rho_filter_gain * Krho_now + (1 - self.K_rho_filter_gain) * self.Krho_est

            # 4. estimate rotation of target
            if not self.use_relative_velocities:
                if self.Final_Phase:
                    angle_to_pred_target = self.Earth_rate * (1000 * self.Theta + current_time)
                else:
                    angle_to_pred_target = self.Earth_rate * (((self.Earth_rate * self.Theta)/V) + current_time)
                    if V <= self.V_MIN:
                        self.use_relative_velocities = True
            else:
                angle_to_pred_target = self.Earth_rate * current_time

            # 5. calculate rotated target vector, crossrange angle, and downrange to go:
            self.target_local_east = np.cross(np.array([0, 0, 1]), self.Target_unit_vector)
            self.Target_unit_vector = self.Target_vector0 / np.linalg.norm(self.Target_vector0) \
                                      + self.Target_unit_vector * (np.cos(angle_to_pred_target) - 1) \
                                      + self.target_local_east * np.sin(angle_to_pred_target)

            self.LATANG = np.dot(self.Target_unit_vector, np.cross(V_I_unit, pos_unit))
            self.Theta = np.arccos(np.dot(self.Target_unit_vector, pos_unit))
            self.downrange_distance = self.Theta * self.ATK

            # Go to selected phase, based on the value of self.Phase:
            # Phase = 1: INITIAL ROLL
            # Phase = 2: HUNTEST
            # Phase = 3: UPCONTROL
            # Phase = 4: BALLISTIC
            # Phase = 5: FINAL

            ###################################################################################
            # Pre-entry attitude hold and initial roll
            ###################################################################################

            if self.Phase == 1:

                if not self.Has_entered:
                    # Phase is pre-entry attitude hold
                    # Check if vehicle has now entered sensible atmosphere, at D > 0.5g
                    aero_g = self.aero_force / 9.81

                    if aero_g >= 0.5:
                        # vehicle has entered sensible atmosphere, set has entered flag to true
                        self.Has_entered = True

                        # Check if velocity is too low
                        if V <= self.v_final:
                            # set L/D to full lift up and set phase to ballistic
                            self.LD_comm = self.MAX_LD
                            self.Phase = 4
                        # check entry angle
                        else:
                            if V >= (self.v_final - (self.K_initial_roll * (self.RDOT/V)**3)):
                                # Entry is too shallow, command full lift down to steepen trajectory
                                self.LD_comm = -1 * self.MAX_LD
                            else:
                                # Entry is steep enough, command full lift up
                                self.LD_comm = self.MAX_LD

                    # LATERAL LOGIC SUBROUTINE

                else:
                    # phase is initial roll
                    # check if drag is too high:
                    if self.aero_force >= self.KA:
                        # Drag is too high for this phase, command full lift up
                        self.LD_comm = self.MAX_LD

                    # Check if altitude rate is shallow enough to proceed to next phase
                    if self.RDOT >= self.RDOT_PHASE2:
                        # altitude rate is low enough to proceed to next phase
                        self.Phase = 2

            ###################################################################################
            # HUNTEST/Constant drag phase
            ###################################################################################

            if self.Phase == 2:
                # Decide which reference L/D to use:
                if self.aero_force >= self.D2:
                    self.ref_LD = self.ref_lD1
                else:
                    self.ref_LD = self.ref_LD2

                # Estimate start of upcontrol phase conditions, check whether altitude is decreasing:
                if self.RDOT <= 0.0:
                    # project conditions to pullout using reference L/D
                    self.V1 = V + self.RDOT / self.ref_LD
                    A0 = ((self.V1/V)**2) * (self.aero_force + ((self.RDOT**2)/(2 * self.HS * self.ref_LD)))
                    A1 = self.aero_force
                else:
                    # project conditions forward using full lift up
                    self.V1 = V + self.RDOT / self.MAX_LD
                    A0 = ((self.V1 / V) ** 2) * (self.aero_force + ((self.RDOT ** 2) / (2 * self.HS * self.MAX_LD)))
                    A1 = A0

                # If first time through the Huntest routine, initialise some variables:
                if self.initial_HUNTEST == False:
                    self.initial_HUNTEST = True
                    Diff_old = 0.0
                    V1_old = self.V1 + self.C18
                    Q7 = Q7F


                Const_drag = False
                Hunting = True

                while Hunting:
                    # Calculate exit velocity at the end of upcontrol
                    self.ALP = 2 * self.C1 * self.HS / (self.ref_LD * self.V1 ** 2)
                    Factor_1 = self.V1 / (1 - self.ALP)
                    Factor_2 = self.ALP * (self.ALP - 1) / A0
                    self.VL = Factor_1 * (1 - np.sqrt(Factor_2 * self.Q7 + self.ALP))

                    # Check whether exit velocity is sufficient to perform skip
                    if self.VL <= self.VLMIN:
                        # Velocity is too low to perform a skip, enter final phase
                        self.Phase = 5
                        self.Final_Phase = True
                        Hunting = False
                    elif self.VL >= self.VSAT:
                        # Skip energy is too excessive, stay in HUNTEST phase, go to constant drag
                        self.Phase = 2
                        Hunting = False
                        Const_drag = True
                    else:
                        VS1 = min(self.V1, VSAT)

                        # Calculate exit flight path angle
                        DVL = VS1 - self.VL
                        DHOOK = (((1 - VS1 / Factor_1) ** 2) - self.ALP) / Factor_2
                        AHOOK = self.CHOOK * ((DHOOK / self.Q7) - 1) / DVL
                        self.gamma_L1 = self.ref_LD * (self.V1 - self.VL) / self.VL
                        self.gamma_L = self.gamma_L1 - (self.CH1 * self.g_scaling * (DVL ** 2) *
                                                        (1 + AHOOK * DVL) / (DHOOK * VL ** 2))

                        # Check if skip can exit atmosphere:
                        if self.gamma_L <= 0.0:
                            # exit not possible, adjust VL and Q7 for conditions at zero flight path angle
                            self.VL = self.VL + (self.gamma_L * self.VL /
                                                 (self.ref_LD - ((3 * AHOOK * (DVL ** 2) + 2 * DVL) *
                                                                 (self.CH1 * self.g_scaling) / (DHOOK * self.VL))))
                            self.Q7 = ((1 - (self.VL / Factor_1) ** 2) - self.ALP) / Factor_2
                            self.gamma_L = 0

                        # calculate simple version of gamma_L, gamma_L1
                        self.gamma_L1 = self.gamma_L1 * (1 - self.Q19) + self.Q19 * self.gamma_L

                        # calculate range to touchdown by adding estimates of each phase
                        # Ballistic phase range
                        self.VBARS = (self.VL / self.VSAT) ** 2
                        self.cos_gamma_L = 1 - (self.gamma_L ** 2) / 2
                        self.ballistic_e = np.sqrt((1 + (self.VBARS - 2) * self.VBARS * self.cos_gamma_L ** 2))
                        self.ASKEP = 2 * self.ATK * \
                                     np.arcsin(self.VBARS * self.cos_gamma_L * self.gamma_L / self.ballistic_e)

                        # Final phase range
                        VL_fps = self.VL / 3.281
                        if VL_fps <= 25262:
                            searching_interpolation = True
                            # use interpolation
                            for i in range(len(self.final_vel_list)):
                                if VL_fps <= self.final_vel_list[i]:
                                    if searching_interpolation:
                                        V_upper = self.final_vel_list[i]
                                        V_lower = self.final_vel_list[i - 1]
                                        searching_interpolation = False

                            upper_list = final_phase_table[str(V_upper)]
                            lower_list = final_phase_table[str(V_lower)]

                            RDOTREF = np.interp(VL_fps, [V_lower, V_upper], [lower_list[0], upper_list[0]])
                            DREFR = np.interp(VL_fps, [V_lower, V_upper], [lower_list[1], upper_list[1]])
                            F2 = np.interp(VL_fps, [V_lower, V_upper], [lower_list[2], upper_list[2]])
                            F1 = np.interp(VL_fps, [V_lower, V_upper], [lower_list[3], upper_list[3]])
                            RTOGO = np.interp(VL_fps, [V_lower, V_upper], [lower_list[4], upper_list[4]])
                            F3 = np.interp(VL_fps, [V_lower, V_upper], [lower_list[5], upper_list[5]])

                            ASPF_nm = RTOGO + F2 * (-1 * self.gamma_L * VL_fps - RDOTREF) + F1 * \
                                      ((self.Q7 / 3.281) - DREFR)
                            self.ASPF = ASPF_nm * 1852

                        else:
                            # use linear about that point
                            self.ASPF = self.Q2 + self.Q3 * self.VL + self.Q5 * (self.Q6 - self.gamma_L)

                        # Upcontrol phase range
                        self.ASPUP = (self.ATK / self.Earth_radius) * (self.HS / self.gamma_L1) * \
                                     np.log((A0 * (self.VL ** 2)) / (self.Q7 * (self.V1 ** 2)))

                        # Downcontrol phase range
                        self.ASPDWN = -1 * self.RDOT * V * self.ATK / (A0 * self.MAX_LD * self.Earth_radius)

                        # Total range
                        self.ASP = self.ASKEP + self.ASPF + self.ASPUP + self.ASPDWN

                        # Difference between predicted and desired downrange
                        Diff = self.downrange_distance - self.ASP

                        if abs(Diff) <= self.TOL:
                            # assuming vehicle remains at current L/D, it will stay within tolerance of target position
                            # switch to upcontrol
                            self.Phase = 3
                            Hunting = False
                        else:
                            if not self.HUNTEST_iteration:
                                # An iteration through HUNTEST has not been performed yet
                                if Diff <= 0.0:
                                    # predicted range is too far, store old values of DIFF and V1
                                    Diff_old = Diff
                                    V1_old = self.V1
                                    Hunting = False
                                    # Go to constant drag
                                    Const_drag = True
                                else:
                                    # predicted range is too close, velocity should be tweaked
                                    V_Corr = self.V1 - V1_old

                                    # Calculate velocity correction
                                    V_Corr = (V_Corr * Diff) / (Diff_old - Diff)

                                    # Limit velocity correction
                                    V_Corr = min(V_Corr, self.V_Corr_lim)

                            # see if exit velocity is increased too much by correction
                            if (self.VSAT - self.VL) <= V_Corr:
                               V_Corr = V_Corr / 2

                            # Apply velocity correction to upcontrol starting velocity
                            self.V1 = self.V1 + V_Corr
                            self.HUNTEST_iteration = True
                            Diff_old = Diff

                # Constant Drag
                if Const_drag:
                    # Calculate Constant Drag
                    self.LD_comm = (-LEQ / self.D0) + self.C16 (self.aero_force - self.D0) - \
                            self.C17 (self.RDOT + 2 * self.HS * self.D0 / V)

                    # Check if commanded L/D will cause too much drag
                    if self.LD_comm <= 0 and self.aero_force >= self.Max_D:
                        # set L/D to neutral
                        self.LD_comm = 0.0


            ###################################################################################
            # Upcontrol/Downcontrol phase
            ###################################################################################

            if self.Phase == 3:
                # Check if high-loft or low-loft
                if not self.high_loft:
                    # low loft algorithm

                    # if velocity is above upcontrol starting velocity, stay in downcontrol
                    if V >= self.V1:
                        # downcontrol still in effect, calcutlate reference altitude rate and drag, then calculate
                        # command L/D based on trajectory errors
                        self.RDOT_ref_DCRTL = self.MAX_LD * (self.V1 - V)
                        self.D_ref_DCTRL = ((V/self.V1)**2) * A0 - ((self.RDOT_ref_DCRTL**2)/(2 * self.HS * self.MAX_LD))
                        self.LD_comm = self.MAX_LD + self.C16 * (self.aero_force - self.D_ref_DCTRL) - \
                                  self.C17 * (self.RDOT - self.RDOT_ref_DCRTL)

                    # check if velocity has reached exit velocity and altitude is decreasing
                    elif V < (self.VL + self.C18) and self.RDOT < 0.0:
                        # move to final phase
                        self.Phase = 5
                        self.Final_Phase = True

                    # check if drag has dropped too low
                    elif self.aero_force <= self.Q7:
                        # switch to ballistic phase
                        self.Phase = 4

                    # check if drag level is still higher than upcontrol starting drag
                    elif self.aero_force >= A0:
                        # decrease drag level, command full lift up
                        self.LD_comm = self.MAX_LD

                        # go to LATLOGIC

                    # if algorithm has run past all tests, proceed to this step
                    else:
                        # RUN PREDGUID
                        self.LD_comm, Diff, velmag_pred, rdot_pred = self.PredGuid()

                        # run NEGTEST
                        if self.LD_comm <= 0 and self.aero_force >= self.Max_D:
                            # set L/D to neutral
                            self.LD_comm = 0.0

                else:
                    # high loft algorithm

                    # check if velocity has reached exit velocity and altitude is decreasing
                    if V < (self.VL + self.C18) and self.RDOT < 0.0:
                        # move to final phase
                        self.Phase = 5
                        self.Final_Phase = True

                    # check if drag has dropped too low
                    elif self.aero_force <= self.Q7:
                        # switch to ballistic phase
                        self.Phase = 4

                    # if algorithm has run past all tests, proceed to this step
                    else:
                        # RUN PREDGUID
                        self.LD_comm, Diff, velmag_pred, rdot_pred = self.PredGuid()

                        # run NEGTEST
                        if self.LD_comm <= 0 and self.aero_force >= self.Max_D:
                            # set L/D to neutral
                            self.LD_comm = 0.0



            ###################################################################################
            # Ballistic phase
            ###################################################################################

            if self.Phase == 4:
                # check if drag has risen high enough
                if self.aero_force >= (self.Q7 + self.KDMIN):
                    # move to final phase
                    self.Phase = 5
                    self.Final_Phase = True
                else:
                    # continue steering with PredGuid, then LATLOGIC
                    self.LD_comm, Diff, velmag_pred, rdot_pred = self.PredGuid()

            ###################################################################################
            # Final Phase
            ###################################################################################

            if self.Phase == 5:
                # check if velocity has dropped below the quit velocity
                if V <= self.VQUIT:
                    # stop steering, command neutral L/D ratio
                    self.LD_comm = 0
                    # create bank angle command
                    RollC = self.K2ROLL * np.arccos((self.LD_comm/self.MAX_LD)) + 2 * np.pi * self.K1ROLL
                    # indicate no more bank angle commands
                    self.stop_bank_comm = True

                else:
                    # check if target has been overshot previously
                    if self.Gone_past:
                        # command full lift down
                        self.LD_comm = -1 * self.MAX_LD

                    # check if target has just now been overshot
                    elif np.dot((np.cross(self.Target_unit_vector, self.pos), np.cross(V_I_unit, pos_unit))) >= 0.0:
                        # target has just now been overshot
                        self.Gone_past = True
                        # command full lift down
                        self.LD_comm = -1 * self.MAX_LD
                    else:
                        # interpolate values based on velocity
                        V_fps = V / 3.281
                        searching_interpolation = True
                        # use interpolation
                        for i in range(len(self.final_vel_list)):
                            if V_fps <= self.final_vel_list[i]:
                                if searching_interpolation:
                                    V_upper = self.final_vel_list[i]
                                    V_lower = self.final_vel_list[i - 1]
                                    searching_interpolation = False

                        upper_list = final_phase_table[str(V_upper)]
                        lower_list = final_phase_table[str(V_lower)]

                        RDOTREF = np.interp(V_fps, [V_lower, V_upper], [lower_list[0], upper_list[0]])
                        DREFR = np.interp(V_fps, [V_lower, V_upper], [lower_list[1], upper_list[1]])
                        F2 = np.interp(V_fps, [V_lower, V_upper], [lower_list[2], upper_list[2]])
                        F1 = np.interp(V_fps, [V_lower, V_upper], [lower_list[3], upper_list[3]])
                        RTOGO = np.interp(V_fps, [V_lower, V_upper], [lower_list[4], upper_list[4]])
                        F3 = np.interp(V_fps, [V_lower, V_upper], [lower_list[5], upper_list[5]])

                        PREDANGL_nm = RTOGO + F2 * (self.RDOT - RDOTREF) + F1 * \
                                      ((self.aero_force / 3.281) - DREFR)
                        self.PREDANGL = PREDANGL_nm * 1852

                        self.LD_comm = self.final_LD + ((4 * (self.downrange_distance - self.PREDANGL))/F3)

                    # check if drag is low enough not to need the G-limiter
                    if self.aero_force >= (self.GMAX / 2):
                        # check if drag has hit the 10g limit
                        if self.aero_force >= self.GMAX:
                            # drag too high, command full lift up
                            self.LD_comm = self.MAX_LD

                            # LATLOGIC
                        else:
                            # calculate limiting altitude rate
                            self.X = np.sqrt(2 * self.HS * (self.GMAX - self.aero_force) * ((LEQ / self.GMAX) + self.MAX_LD)
                                             + ((2 * self.HS * (self.GMAX/V))**2))

                            # check if altitude rate exceeds the limiting altitude rate
                            if self.RDOT <= (-1 * self.X):
                                # g-loads too high, command full lift up
                                self.LD_comm = self.MAX_LD

            ###################################################################################
            # Lateral logic
            ###################################################################################

            # check if quit steering command has been given
            if not self.stop_bank_comm:
                # check if target has been overshot
                if not self.Gone_past:
                    # vehicle will not immediately be commanding full lift down, lateral control available

                    # calculate lateral switch limit
                    self.Y  = self.KLAT * VSQ + (self.LATBIAS / self.ATK)

                    # check if L/D command is within 15 degrees of full lift up or full lift down
                    if abs(self.LD_comm) >= self.LDCMIR:
                        # lateral control authority of vehicle is reduced, reduce lateral switch by half
                        self.Y = self.Y / 2

                        # check whether lift vector being in this quadrant is causing the lateral angle to decrease
                        if (self.K2ROLL * self.LATANG) >= 0.0:
                            # increase the amount of control in the lateral channel by setting the L/D command to 15 deg
                            self.LD_comm = self.LDCMIR * np.sign(self.LD_comm)
                            # command to not perform a bank reversal
                            self.skip_bank_reversal = True

                    # check if lateral angle limit is exceeded
                    if (self.K2ROLL * self.LATANG) >= self.Y and not self.skip_bank_reversal:
                        # command bank reversal by switching the quadrant of the bank angle
                        self.K2ROLL = -1 * self.K2ROLL
                        # check whether the vehicle is in a lift down configuration
                        if self.LD_comm <= 0.0:
                            # command roll reversal through lift down
                            # increment the revolution counter
                            self.K1ROLL = self.K1ROLL - self.K2ROLL

                self.skip_bank_reversal = False

                # check whether commanded L/D is larger than vehicle max L/D
                if abs(self.LD_comm) >= self.MAX_LD:
                    # command vehicle max L/D
                    self.LD_comm = np.sign(self.LD_comm) * self.MAX_LD

                # calculate bank angle command from the desired L/D

                RollC = self.K2ROLL * np.arccos(self.LD_comm / self.MAX_LD) + 2 * np.pi * self.K1ROLL

            # update bank angle
            self.bank_angle = RollC
            # update guidance time
            self.current_time = current_time


class STSAerodynamicGuidance:

    def __init__(self, bodies: environment.SystemOfBodies):

        # Extract the STS and Earth bodies
        self.vehicle = bodies.get_body("Capsule")
        self.earth = bodies.get_body("Earth")

        # Extract the STS flight conditions, angle calculator, and aerodynamic coefficient interface
        environment_setup.add_flight_conditions( bodies, 'Capsule', 'Earth' )
        self.vehicle_flight_conditions = bodies.get_body("Capsule").flight_conditions
        self.aerodynamic_angle_calculator = self.vehicle_flight_conditions.aerodynamic_angle_calculator
        self.aerodynamic_coefficient_interface = self.vehicle_flight_conditions.aerodynamic_coefficient_interface

        self.current_time = float("NaN")

    def getAerodynamicAngles(self, current_time: float):
        self.updateGuidance( current_time )
        return np.array([self.angle_of_attack, 0.0, self.bank_angle])

    # Function that is called at each simulation time step to update the ideal bank angle of the vehicle
    def updateGuidance(self, current_time: float):
        print('guidance')

        if( math.isnan( current_time ) ):
            self.current_time = float("NaN")
        elif( current_time != self.current_time ):
            # Get the (constant) angular velocity of the Earth body
            earth_angular_velocity = np.linalg.norm(self.earth.body_fixed_angular_velocity)
            # Get the distance between the vehicle and the Earth bodies
            earth_distance = np.linalg.norm(self.vehicle.position)
            # Get the (constant) mass of the vehicle body
            body_mass = self.vehicle.mass

            # Extract the current Mach number, airspeed, and air density from the flight conditions
            mach_number = self.vehicle_flight_conditions.mach_number
            airspeed = self.vehicle_flight_conditions.airspeed
            density = self.vehicle_flight_conditions.density

            # Set the current Angle of Attack (AoA). The following line enforces the followings:
            # * the AoA is constant at 40deg when the Mach number is above 12
            # * the AoA is constant at 10deg when the Mach number is below 6
            # * the AoA varies close to linearly when the Mach number is between 12 and 6
            # * a Logistic relation is used so that the transition in AoA between M=12 and M=6 is smoother
            self.angle_of_attack = np.deg2rad(30 / (1 + np.exp(-2*(mach_number-9))) + 10)

            # Update the variables on which the aerodynamic coefficients are based (AoA and Mach)
            current_aerodynamics_independent_variables = [self.angle_of_attack, mach_number]

            # Update the aerodynamic coefficients
            self.aerodynamic_coefficient_interface.update_coefficients(
                current_aerodynamics_independent_variables, current_time)

            # Extract the current force coefficients (in order: C_D, C_S, C_L)
            current_force_coefficients = self.aerodynamic_coefficient_interface.current_force_coefficients
            # Extract the (constant) reference area of the vehicle
            aerodynamic_reference_area = self.aerodynamic_coefficient_interface.reference_area

            # Get the heading, flight path, and latitude angles from the aerodynamic angle calculator
            heading = self.aerodynamic_angle_calculator.get_angle(environment_setup.aerodynamic_coefficients.AerodynamicsReferenceFrameAngles.heading_angle)
            flight_path_angle = self.aerodynamic_angle_calculator.get_angle(environment_setup.aerodynamic_coefficients.AerodynamicsReferenceFrameAngles.flight_path_angle)
            latitude = self.aerodynamic_angle_calculator.get_angle(environment_setup.aerodynamic_coefficients.AerodynamicsReferenceFrameAngles.latitude_angle)

            # Compute the acceleration caused by Lift
            lift_acceleration = 0.5 * density * airspeed ** 2 * aerodynamic_reference_area * current_force_coefficients[2] / body_mass
            # Compute the gravitational acceleration
            downward_gravitational_acceleration = self.earth.gravitational_parameter / (earth_distance ** 2)
            # Compute the centrifugal acceleration
            spacecraft_centrifugal_acceleration = airspeed ** 2 / earth_distance
            # Compute the Coriolis acceleration
            coriolis_acceleration = 2 * earth_angular_velocity * airspeed * np.cos(latitude) * np.sin(heading)
            # Compute the centrifugal acceleration from the Earth
            earth_centrifugal_acceleration = earth_angular_velocity ** 2 * earth_distance * np.cos(latitude) *             (np.cos(latitude) * np.cos(flight_path_angle) + np.sin(flight_path_angle) * np.sin(latitude) * np.cos(heading))

            # Compute the cosine of the ideal bank angle
            cosine_of_bank_angle = ((downward_gravitational_acceleration - spacecraft_centrifugal_acceleration) * np.cos(flight_path_angle) - coriolis_acceleration - earth_centrifugal_acceleration) / lift_acceleration
            # If the cosine lead to a value out of the [-1, 1] range, set to bank angle to 0deg or 180deg
            if (cosine_of_bank_angle < -1):
                self.bank_angle = np.pi
            elif (cosine_of_bank_angle > 1):
                self.bank_angle = 0.0
            else:
                # If the cos is in the correct range, return the computed bank angle
                self.bank_angle = np.arccos(cosine_of_bank_angle)
            self.current_time = current_time