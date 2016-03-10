attributes = ['velocities']
required = ['momenta', 'topology']

def velocities(snapshot):
    """
    Returns
    -------
    velocites : numpy.ndarray
        velocities
    """
    return snapshot.momenta * snapshot.topology.inverse_masses
    
