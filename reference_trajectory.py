import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos, exp

from scipy.integrate import solve_ivp

from typing import Callable

# Fix Python 3's weird rounding function
# https://stackoverflow.com/a/44888699/538379
round2=lambda x,y=None: round(x+1e-15,y)


def reference_bank_angle(t, X, params):
    """Generates the reference bank angle profile"""
    v = X[2]
    return np.deg2rad(20.0)


def traj_eom(t: float,
             state: np.array,
             params: dict,
             bank_angle_fn: Callable[[float, np.array, dict], float]
             ):

    h, s, v, gam = state
    u = bank_angle_fn(t, state, params)

    rho0 = params['rho0']
    H = params['H']
    beta = params['beta']  # m/(Cd * Aref)
    LD = params['LD']
    R_m = params['R_m']
    g = params['g']

    v2 = v * v
    rho = rho0 * exp(-h / H)
    D_m = rho * v2 / (2 * beta)  # Drag Acceleration (D/m)
    r = R_m + h
    return np.array([v * sin(gam),  # dh/dt
                     v * cos(gam),  # ds/dt
                     -D_m - g * sin(gam),  # dV/dt
                     (v2 * cos(gam) / r + D_m * LD * cos(u) - g * cos(gam)) / v]  # dgam/dt
                    )


def traj_eom_with_costates(t: float,
                           state: np.array,
                           params: dict,
                           bank_angle_fn: Callable[[float, np.array, dict], float]
                           ):
    lamS = 1
    h, s, V, gam, lamH, lamV, lamGAM, lamU = state

    u = bank_angle_fn(t, state, params)

    rho0 = params['rho0']
    H = params['H']
    beta = params['beta']
    LD = params['LD']
    R_m = params['R_m']
    g = params['g']

    r = R_m + h

    v = V
    V2 = V * V
    rho = rho0 * exp(-h / H)
    D_m = rho * V2 / (2 * beta)  # Drag Acceleration (D/m)

    #     lamHDot = D_m*LD*lamGAM*cos(u)/(H*v) - D_m*lamV/H + lamGAM*v*cos(gam)/r**2
    #     lamVDot = D_m*LD*lamGAM*cos(u)/v**2 - LD*lamGAM*rho*cos(u)/beta - g*lamGAM*cos(gam)/v**2 - lamGAM*cos(gam)/r \
    #               - lamH*sin(gam) \
    #               - lamS*cos(gam) \
    #               + lamV*rho*v/beta
    #     lamGAMDot = g*lamV*cos(gam) - lamGAM*(g*sin(gam) - v**2*sin(gam)/r)/v - lamH*v*cos(gam) + lamS*v*sin(gam)

    lamHdot = D_m * LD * lamGAM * cos(u) / (H * v) - D_m * lamV / H + lamGAM * v * cos(gam) / r ** 2
    lamVdot = D_m * LD * lamGAM * cos(u) / v ** 2 - LD * lamGAM * rho * cos(u) / beta - g * lamGAM * cos(
        gam) / v ** 2 - lamGAM * cos(gam) / r - lamH * sin(gam) - lamS * cos(gam) + lamV * rho * v / beta
    lamGAMdot = -g * lamGAM * sin(gam) / v + g * lamV * cos(gam) + lamGAM * v * sin(gam) / r - lamH * v * cos(
        gam) + lamS * v * sin(gam)

    #     lamUdot = -LD*lamGAM*rho*v*sin(u)/(2*beta)
    lamUdot = LD * lamGAM * rho * v * sin(u) / (2 * beta)
    return np.array([V * sin(gam),  # dh/dt
                     V * cos(gam),  # ds/dt
                     -D_m - g * sin(gam),  # dV/dt
                     (V2 * cos(gam) / r + D_m * LD * cos(u) - g * cos(gam)) / V,  # dgam/dt

                     lamHdot,
                     lamVdot,
                     lamGAMdot,
                     lamUdot]
                    )

class Trajectory:
    """Data structure for holding the result of a simulation run"""

    def __init__(self, t: float, X: np.array, u: np.array, params: dict):
        self.t = t
        self.X = X
        self.u = u
        self.params = params


def simulate_entry_trajectory(eom: Callable[[float, np.array], np.array],
                              t0: float,
                              tf: float,
                              X0: np.array,
                              term_var_idx: int,
                              term_var_val: float,
                              params: dict,
                              bank_angle_fn: Callable[[float, np.array, dict], float],
                              t_eval: [np.array] = None) -> Trajectory:
    altitude_stop_event = lambda t, X, params, _: X[term_var_idx] - term_var_val
    altitude_stop_event.terminal = True
    altitude_stop_event.direction = -1

    output = solve_ivp(eom,
                       [t0, tf],
                       X0,
                       args=(params, bank_angle_fn),
                       t_eval=t_eval,
                       rtol=1e-6, events=altitude_stop_event)

    # loop over output and compute bank angle for each timestep
    num_steps = len(output.t)

    u = np.zeros(num_steps)
    for i, (t, X) in enumerate(zip(output.t, output.y.T)):
        u[i] = bank_angle_fn(t, X, params)

    # Transpose y so that each state is in a separate column and each row
    # represents a timestep
    return Trajectory(output.t, output.y.T, u, params)


class ApolloReferenceData:
    def __init__(self, X_and_lam: np.array, u: np.array, tspan: np.array, params: dict):
        """
        X_and_lam: [h, s, v, gam, lamH, lamV, lamGAM, lamU] - 8 x n matrix
        tspan: 1 x n vector
        """
        self.X_and_lam = X_and_lam
        self.tspan = tspan
        self.params = params
        self.u = u

        assert len(X_and_lam.shape) == 2 and X_and_lam.shape[0] > 1, "Need at least two rows of data"
        self.num_rows = X_and_lam.shape[0]

        self.delta_v = abs(X_and_lam[1, 2] - X_and_lam[0, 2])
        assert self.delta_v > 0, "Reference trajectory has repeated velocites in different rows"

        self.start_v = X_and_lam[0, 2]

        F1, F2, F3, D_m, hdot_ref = self._compute_gains_and_ref()
        F3[-1] = F3[-2]  # Account for F3=0 at t=tf
        # Stack the columns as follows:
        # [t, h, s, v, gam, F1, F2, F3, D/m]
        self.data = np.column_stack((tspan, X_and_lam[:, :4], F1, F2, F3, D_m, hdot_ref))

    def _compute_gains_and_ref(self):
        h = self.X_and_lam[:, 0]
        v = self.X_and_lam[:, 2]
        gam = self.X_and_lam[:, 3]

        lamH = self.X_and_lam[:, 4]
        lamGAM = self.X_and_lam[:, 6]
        lamU = self.X_and_lam[:, 7]

        rho0 = self.params['rho0']
        H = self.params['H']
        beta = self.params['beta']  # m/(Cd * Aref)

        v2 = v * v
        rho = rho0 * exp(-h / H)
        D_m = rho * v2 / (2 * beta)  # Drag Acceleration (D/m)
        hdot = v * sin(gam)

        F1 = H * lamH / D_m
        F2 = lamGAM / (v * np.cos(gam))
        F3 = lamU
        return F1, F2, F3, D_m, hdot

    def get_row_by_velocity(self, v: float):
        """
        Returns data row closest to given velocity
        """
        all_v = self.data[:, 3]
        dist_to_v = np.abs(all_v - v)
        index = min(dist_to_v) == dist_to_v
        return self.data[index, :][0]

    def save(self, filename: str):
        """Saves the reference trajectory data to a file"""
        np.savez(filename, X_and_lam=self.X_and_lam, u=self.u, tspan=self.tspan, params=self.params)

    @staticmethod
    def load(filename: str):
        """Initializes a new ApolloReferenceData from a saved data file"""
        npzdata = np.load(filename, allow_pickle=True)
        X_and_lam = npzdata.get('X_and_lam')
        u = npzdata.get('u')
        tspan = npzdata.get('tspan')
        params = npzdata.get('params').item()
        return ApolloReferenceData(X_and_lam, u, tspan, params)

# Initial conditions
h0 = 79.4E3 # Entry altitude
V0 = 7003  # Entry velocity
gamma0_deg = -3.21 # Entry flight path angle
s0 = 0

# Model params
params = {'H': 7200,
              'rho0': 1.225, # kg/m^3
              'beta': 246.7,
              'LD': 0.26,
              'R_m': 6371e3,
              'g': 9.81}

# Terminal altitude
h_f = 30000

gamma0 = np.deg2rad(gamma0_deg)
X0 = np.array([h0, s0, V0, gamma0])
t0 = 0
tf = 320
tspan = np.linspace(t0, tf, 1001)

ref_traj = simulate_entry_trajectory(traj_eom, t0, tf, X0, 0, h_f, params, reference_bank_angle, tspan)

plt.plot(ref_traj.X[:,2]/1e3, ref_traj.X[:,0]/1e3)
plt.xlabel('V [km/s]')
plt.ylabel('h [km]')
plt.grid(True)
plt.show()

ref_tf = ref_traj.t[-1]
ref_tspan_rev = ref_traj.t[::-1] # Reverse the time span
Xf = np.copy(ref_traj.X[-1,:])

# Ensure monotonic decreasing V
def V_event(t,X,p,_):
    return X[3] - 5500

V_event.direction = 1
V_event.terminal = True

X_and_lam0 = np.concatenate((Xf, [-1/np.tan(Xf[3]), 0, 0, 0]))
output = solve_ivp(traj_eom_with_costates, # lambda t,X,p,u: -traj_eom_with_costates(t,X,p,u),
                   [ref_tf, 0],
                   X_and_lam0,
                   t_eval=ref_traj.t[::-1],
                   rtol=1e-8,
                   events=V_event,
                   args=(params, reference_bank_angle))
lam = output.y.T[:,4:][::-1]
X_and_lam = output.y.T[::-1]

np.set_printoptions(suppress=True)

# Test loading and saving of data
apollo_ref = ApolloReferenceData(X_and_lam, ref_traj.u, ref_traj.t, params)
apollo_ref.save('apollo_data_vref.npz')
print("gamma[0] =", X_and_lam[0,3])
print("gamma[0] in deg =", np.rad2deg(X_and_lam[0,3]))

# Load data back and check that it matches the original
ref = ApolloReferenceData.load('apollo_data_vref.npz')
assert np.allclose(ref.data, apollo_ref.data)