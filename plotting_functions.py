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