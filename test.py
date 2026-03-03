import numpy as np
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel.interface import spice
from tudatpy.kernel.numerical_simulation import environment, environment_setup

a = [ 0.82656686, -0.52517494,  0.20243151]
b = [ 0.89371342, -0.36852935,  0.25585631]
c = np.cross(a,b)
print(c)