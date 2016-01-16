
class MMSTHamiltonian(PotentialEnergySurface):
    """Meyer-Miller-Stock-Thoss mapped electron Hamiltonian

    Parameters
    ----------
    H_matrix : matrix-like
        The input Hamiltonian matrix. By "matrix-like", it is meant that it
        must have the following properties
    """

    def __init__(self, H_matrix):
        pass

    def H(self, snap):
        pass

    def V(self, snap):
        pass
    
    def dHdq(self, snap):
        pass

    def dHdp(self, snap):
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

