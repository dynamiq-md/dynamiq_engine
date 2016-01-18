import dynamiq_engine as dynq
import numpy as np

class PotentialStub(dynq.potentials.PotentialEnergySurface):
    def __init__(self):
        self.n_atoms = 1
        self.n_spatial = 1

    def H(self, snapshot):
        return 0.0

    def dHdq(self, snapshot):
        return np.array([0.0])

    def dHdp(self, snapshot):
        return np.array([0.0])
