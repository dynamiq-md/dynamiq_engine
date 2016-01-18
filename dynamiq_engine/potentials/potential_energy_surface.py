import numpy as np

class PotentialEnergySurface(object):
    """Abstract class for potential energy surfaces

    Attributes
    ----------
    dynamics_level : integer (0, 1, or 2)
        Level required to simulate the system
    """
    def H(self, snapshot):
        raise NotImplementedError("Using generic PES object")

    def kinetic_energy(self, snapshot):
        return 0.5*np.dot(snapshot.velocities, snapshot.momenta)

    def dHdq(self, snapshot):
        raise NotImplementedError("Using generic PES object")

    def dHdp(self, snapshot):
        raise NotImplementedError("Using generic PES object")

    def d2Hdq2(self, snapshot):
        raise NotImplementedError("Using generic PES object")

    def d2Hdp2(self, snapshot):
        raise NotImplementedError("Using generic PES object")

    def d2Hdqdp(self, snapshot):
        raise NotImplementedError("Using generic PES object")

    def d2Hdpdq(self, snapshot):
        raise NotImplementedError("Using generic PES object")

class OneDimensionalInteractionModel(PotentialEnergySurface):
    def __init__(self, interaction):
        super(OneDimensionalInteractionModel, self).__init__()
        self.n_atoms = 1
        self.n_spatial = 1
        self.dynamics_level = 0
        self.interaction = interaction

    def H(self, snapshot):
        x = snapshot.coordinates[0]
        return self.kinetic_energy(snapshot) + self.interaction.f(x)

    def dHdq(self, snapshot):
        x = snapshot.coordinates[0]
        return np.array([self.interaction.dfdx(x)])

    def dHdp(self, snapshot):
        return np.array([snapshot.velocities[0]])

    # TODO: the rest is only necessary for full semiclassical calculations
