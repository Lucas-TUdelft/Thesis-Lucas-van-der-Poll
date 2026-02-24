import numpy as np
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel.interface import spice
from tudatpy.kernel.numerical_simulation import environment, environment_setup

# 1. Load spice kernels (needed for the rotation matrix)
spice.load_standard_kernels()

# 2. Define your parameters
epoch = 0.0  # J2000 epoch (change this to your simulation_start_epoch)
lat_in = np.deg2rad(14.9198) # in rad
lon_in = np.deg2rad(-23.5073) # in rad
alt_in = 30.0 # in meters
R_earth = 6378137.0 # in meters

# 3. Convert to Body-Fixed Cartesian
cartesian_itrs = element_conversion.spherical_to_cartesian_elementwise(
    R_earth + alt_in, lat_in, lon_in, 0, 0, 0)[:3]

body_settings = environment_setup.get_default_body_settings(
    ['Earth'], 'SSB', 'J2000')

# Precise rotation model from Geocentric to International Celestial Reference Frame
body_settings.add_empty_settings("Earth")
body_settings.get('Earth').rotation_model_settings = environment_setup.rotation_model.gcrs_to_itrs(
    base_frame='J2000')

# Create the system of bodies
bodies = environment_setup.create_system_of_bodies(body_settings)

# Get the Rotation Matrix from body-fixed frame to inertial rotation
earth_rotation_model = bodies.get_body('Earth').rotation_model
rotation_matrix = earth_rotation_model.body_fixed_to_inertial_rotation(epoch)

# Transform the vector
cartesian_j2000 = np.dot(rotation_matrix, cartesian_itrs)
# Pad the 3-element position with three 0.0s for velocity
cartesian_state_6d = np.append(cartesian_j2000, [0.0, 0.0, 0.0])

# This will be the inertial cartesian state of station, but the want station coords are given in body-fixed frame...
cartesian_state_inertial = environment.transform_to_inertial_orientation(
    np.append(cartesian_itrs, [0.0, 0.0, 0.0]),
    epoch,
    bodies.get_body('Earth').rotation_model
)

#...so we convert back to body-fixed frame using the inverse rotation matrix
cartesian_body_fixed = np.dot(np.linalg.inv(rotation_matrix), cartesian_state_inertial[:3])
cartesian_body_fixed_6d = np.append(cartesian_body_fixed, [0.0, 0.0, 0.0])

# Now call the function
reverted_spherical = element_conversion.cartesian_to_spherical(cartesian_state_6d) # this will br wrong for comparison
reverted_spherical_rotated = element_conversion.cartesian_to_spherical(cartesian_body_fixed_6d) # this is correct for comparison

# Extract Lat/Long
lat = reverted_spherical[1]
long = reverted_spherical[2]

print(f"Original Lon: {np.rad2deg(lon_in):.4f}")
print(f"Body-Fixed Lon: {np.rad2deg(reverted_spherical_rotated[2]):.4f}")
print(f"Difference (correct):   {np.rad2deg(reverted_spherical_rotated[2] - lon_in):.4f} degrees")
print(f"Inertial Lon: {np.rad2deg(reverted_spherical[2]):.4f}")
print(f"Difference (wrong, comparing apples to pears):   {np.rad2deg(reverted_spherical[2] - lon_in):.4f} degrees")