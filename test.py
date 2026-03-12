import numpy as np
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel.interface import spice
from tudatpy.kernel.numerical_simulation import environment, environment_setup

capsule_initial_mass = 19483.234917039015
rho0 = 1.225  # kg/m^3
H = 7200  # m
h = 30000  # m
v = -670  # m/s
dt = 0.1
t = 0

descending = True
while descending:

    rho = rho0 * np.exp(-h / H)

    Fg = 9.81 * capsule_initial_mass
    Fd = 0.5 * 1.15 * rho * (v ** 2) * (np.pi * (2.7 ** 2))

    a = (Fd - Fg) / capsule_initial_mass
    v = v + a * dt
    print(v)
    h = h + v * dt

    t = t + dt

    if t >= 10:
        descending = False