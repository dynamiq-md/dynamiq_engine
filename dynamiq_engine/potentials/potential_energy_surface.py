class PotentialEnergySurface(object):
    """Abstract class for potential energy surfaces

    Attributes
    ----------
    derivatives_defined : integer (0, 1, or 2)
        Level required to simulate the system
    """
    def __init__(self, n_dim):
        self.derivatives_defined = 0

    def H(self, snapshot):
        raise NotImplementedError("Using generic PES object")

    def T(self, snapshot):
        pass

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

class OneDimensionalModel(PotentialEnergySurface):
    def __init__(self, interaction):
        self.derivatives_defined = 2
        self.interaction = interaction

    def H(self, snapshot):
        x = snapshot.positions[0]
        return self.kinetic_energy(snapshot) + self.interaction.f(x)

