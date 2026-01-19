import numpy as np
import matplotlib.pyplot as plt
import os

lookup_tables_path = os.path.join(os.getcwd(),"AerodynamicLookupTables")
CD_path = os.path.join(lookup_tables_path,"CD_table.txt")
CL_path = os.path.join(lookup_tables_path, "CL_table.txt")

with open(CD_path, 'r') as f:
    CD_lines = [line.strip() for line in f if line.strip()]
    f.close()

with open(CL_path, 'r') as f:
    CL_lines = [line.strip() for line in f if line.strip()]
    f.close()

altitudes = np.array([float(x) for x in CD_lines[1].split()])
mach_numbers = np.array([float(x) for x in CL_lines[2].split()])

n_alt = len(altitudes)
n_mach = len(mach_numbers)

CD_coeff = np.zeros((n_alt, n_mach))
CL_coeff = np.zeros((n_alt, n_mach))

for i in range(n_alt):
    CD_coeff[i,:] = np.array([float(x) for x in CD_lines[3 + i].split()])
    CL_coeff[i,:] = np.array([float(x) for x in CL_lines[3 + i].split()])

plt.figure()
for i in range(n_alt):
    plt.plot(mach_numbers, CD_coeff[i,:], label = str(altitudes[i]))

plt.xlabel("Mach number [-]")
plt.ylabel("CD [-]")
plt.legend()
plt.grid(True)
plt.show()

for i in range(n_alt):
    plt.plot(mach_numbers, CL_coeff[i, :], label=str(altitudes[i]))

plt.xlabel("Mach number [-]")
plt.ylabel("CL [-]")
plt.legend()
plt.grid(True)
plt.show()

for i in range(n_alt):
    plt.plot(mach_numbers, CL_coeff[i, :]/CD_coeff[i, :], label=str(altitudes[i]))

plt.xlabel("Mach number [-]")
plt.ylabel("CL/CD [-]")
plt.legend()
plt.grid(True)
plt.show()


plt.figure()
for i in range(n_mach):
    plt.plot(altitudes, CD_coeff[:,i], label = str(mach_numbers[i]))

plt.xlabel("altitude [km]")
plt.ylabel("CD [-]")
plt.legend()
plt.grid(True)
plt.show()

for i in range(n_mach):
    plt.plot(altitudes, CL_coeff[:, i], label=str(mach_numbers[i]))

plt.xlabel("altitude [km]")
plt.ylabel("CL [-]")
plt.legend()
plt.grid(True)
plt.show()

for i in range(n_mach):
    plt.plot(altitudes, CL_coeff[:, i]/CD_coeff[:, i], label=str(mach_numbers[i]))

plt.xlabel("altitude [km]")
plt.ylabel("CL/CD [-]")
plt.legend()
plt.grid(True)
plt.show()