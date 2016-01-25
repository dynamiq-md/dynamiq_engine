import dynamiq_engine as dynq
from dynamiq_engine.potentials import PotentialEnergySurface
import numpy as np

class MMSTHamiltonian(PotentialEnergySurface):
    """Meyer-Miller-Stock-Thoss mapped electron Hamiltonian.

    Parameters
    ----------
    H_matrix : dynamiq_engine.NonadiabaticMatrix
        The input Hamiltonian matrix. 
    """

    def __init__(self, H_matrix, electronic_first=True):
        self.H_matrix = H_matrix
        self.n_electronic_states = H_matrix.n_electronic_states
        self.electronic_first = electronic_first
        pass

    def H(self, snap):
        pass

    def V(self, snap):
        pass
    
    def set_electronic_dHdq(self, electronic_dHdq, snapshot):
        pass

    def set_dHdq(self, dHdq, snapshot):
        pass

    def set_electronic_dHdp(self, electronic_dHdq, snapshot):
        pass

    def set_dHdp(self, dHdq, snapshot):
        pass

    # following are to be done later
    def L(self, snap):
        pass

    def d2Hdq2(self, snap):
        pass

    def d2Hdp2(self, snap):
        pass

    def d2Hdqdp(self, snap):
        pass

    def d2Hdpdq(self, snap):
        pass

