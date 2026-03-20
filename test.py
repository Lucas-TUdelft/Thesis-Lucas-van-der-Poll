import numpy as np
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel.interface import spice
from tudatpy.kernel.numerical_simulation import environment, environment_setup

deadband = 2.0
heading_error = -.0

if heading_error >= deadband:
    bank_sign = 1.0
elif heading_error <= deadband:
    bank_sign = -1.0

print(bank_sign)