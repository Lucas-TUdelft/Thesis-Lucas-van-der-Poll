import numpy as np
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel.interface import spice
from tudatpy.kernel.numerical_simulation import environment, environment_setup

width_needed = 155
hight_needed = 190

width = 20
hight = 10
cost = 13.0

n_width = (width_needed // width)
if width_needed % width != 0:
    n_width += 1
n_hight = (hight_needed // hight)
if hight_needed % hight != 0:
    n_hight += 1

print('total cost:', n_width * n_hight * cost)