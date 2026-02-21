import matplotlib.pyplot as plt

def maximum_error_plot(time_steps,max_errors):
    '''
    Plot the maximum position error

    Parmeters
    ----------
    time_steps : list
        benchmark time steps used
    max_errors : list
        maximum error attained in each of the used benchmarks

    Returns
    ----------
    none
    '''

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111)

    ax1.loglog(time_steps,max_errors)
    ax1.scatter(time_steps,max_errors)
    ax1.set_xlabel('time step [s]')
    ax1.set_ylabel('maximum error [m]')
    ax1.grid()

    fig.tight_layout()
    plt.show()

    return

def altitude_plot(h,t):
    '''
    plot the altitude over time for the capsule

    Parmeters
    ----------
    h : list
        altitude at each time step
    t : list
        time

    Returns
    ----------
    none
    '''

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111)
    ax1.plot(t,h)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Altitude [m]')
    ax1.grid()
    fig.tight_layout()
    plt.show()

    return

def velocity_plot(v,t):
    '''
    plot the velocity over time for the capsule

    Parmeters
    ----------
    v : list
        velocity at each time step
    t : list
        time

    Returns
    ----------
    none
    '''

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111)
    ax1.plot(t,v)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Velocity [m/s]')
    ax1.grid()
    fig.tight_layout()
    plt.show()

    return

def gload_plot(g,t):
    '''
    plot the velocity over time for the capsule

    Parmeters
    ----------
    g : list
        aerodynamic g-load at each time step
    t : list
        time

    Returns
    ----------
    none
    '''

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111)
    ax1.plot(t,g)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Velocity [m/s]')
    ax1.grid()
    fig.tight_layout()
    plt.show()

    return

def latlong_plot(lat, long, lat_groundstation, long_groundstation):
    '''
    plot the latitude and longitude for the capsule, and for the ground station

    Parmeters
    ----------
    lat : list
        latitude at each time step
    long : list
        longitude at each time step
    lat_groundstation : float
        latitude of the ground station
    long_groundstation : float
        longitude of the ground

    Returns
    ----------
    none
    '''

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111)
    ax1.scatter(long, lat, s=1)
    ax1.plot(long_groundstation, lat_groundstation, color = 'g', marker = '*')
    ax1.set_xlabel('latitude [deg]')
    ax1.set_ylabel('longitude [deg]')
    ax1.grid()
    fig.tight_layout()
    plt.show()

    return

def bank_plot(bank_angle,t):
    '''
    plot the bank angle over time for the capsule

    Parmeters
    ----------
    h : list
        bank angle at each time step
    t : list
        time

    Returns
    ----------
    none
    '''

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111)
    ax1.plot(t,bank_angle)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Bank Angle [deg]')
    ax1.grid()
    fig.tight_layout()
    plt.show()

    return

def unprocessed_state_plot(t,states):
    ''' Plot the unprocessed states of a propagator '''

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111)

    for i in range(len(states[0]) - 1):
        label = i + 1
        state_element = states[:,i + 1]
        ax1.plot(t, state_element, label = label)
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Position error w.r.t. benchmark [m]')
        ax1.grid(True, which='both', ls="-")
        fig.tight_layout()
    ax1.legend()
    plt.show()

    return

def propagator_error_plot(t,e,labels):
    ''' Plot the magnitude of the position error as a function of time, for all propagators '''

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111)
    linestyles = ['-','--','-',':','-','--',':']

    for i in range(len(labels)):
        label = labels[i]
        ax1.plot(t[i], e[i], label=label, linestyle = linestyles[i])
        ax1.semilogy()
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Position error w.r.t. benchmark [m]')
        ax1.grid(True,which='both',ls="-")
        ax1.legend()

    fig.tight_layout()
    plt.show()

    return

def altitude_comparison_plot(altitudes,times,labels):
    ''' plot the altitude over time of various propagators '''

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111)

    linestyles = ['-','-','--',':','--',':',':','--',':',':']

    for i in range(len(altitudes)):
        ax1.plot(times[i], altitudes[i],label = labels[i], linestyle = linestyles[i])

    ax1.set_xlabel('Time [s]', size=16)
    ax1.set_ylabel('Altitude [m]', size=16)
    ax1.grid(True, which='both', ls="-")
    ax1.legend()
    fig.tight_layout()
    plt.show()

    return

def aero_comparison_plot(aeros,times,labels):
    ''' plot the aero-g load over time of various propagators '''

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111)

    linestyles = ['-','-','--',':','--',':',':','--',':',':']

    for i in range(len(aeros)):
        ax1.plot(times[i], aeros[i],label = labels[i], linestyle = linestyles[i])

    ax1.set_xlabel('Time [s]', size=16)
    ax1.set_ylabel('Aerodynamic g-load [-]', size=16)
    ax1.grid(True, which='both', ls="-")
    ax1.legend()
    fig.tight_layout()
    plt.show()

    return

def error_plot(t,e,timesteps):
    ''' Plot the magnitude of the position error as a function of time, for all time steps '''

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111)

    for i in range(len(timesteps)):
        label = 'ε = ' + str(timesteps[i])
        ax1.plot(t[i], e[i], label=label)
        ax1.semilogy()
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('|| ε_r || [m]')
        ax1.grid(True,which='both',ls="-")
        ax1.legend()

    fig.tight_layout()
    plt.show()

    return

def time_steps_plot(timesteps,t):
    ''' plot the time steps taken over time for the capsule '''

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111)
    ax1.plot(t,timesteps)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Time Step [s]')
    ax1.grid()
    fig.tight_layout()
    plt.show()

    return

def integrator_propagator_plot(evaluation_numbers,maximum_errors,labels):
    ''' Plot the maximum error against the number of function evaluations,
     for each integrator & propagator combination '''

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111)

    for i in range(len(evaluation_numbers)):
        ax1.loglog(evaluation_numbers[i],maximum_errors[i],label = labels[i])
        ax1.scatter(evaluation_numbers[i],maximum_errors[i])

    ax1.set_xlabel('Number of Function Evaluations [-]', size=16)
    ax1.set_ylabel('Maximum Error [m]', size=16)
    ax1.grid(True, which='both', ls="-")
    ax1.legend()
    fig.tight_layout()
    plt.show()

    return

def state_difference_plot(differences,times,labels):
    ''' plot the difference in states between a two propagations '''

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111)

    for i in range(len(differences)):
        ax1.plot(times[i], differences[i], label = labels[i])

    #ax1.semilogy()
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('position error [m]')
    ax1.grid()
    ax1.legend()

    fig.tight_layout()
    plt.show()
    return