import numpy as np

class PotentialEnergySurface(object):
    """Abstract class for potential energy surfaces

    Attributes
    ----------
    dynamics_level : integer (0, 1, or 2)
        Level required to simulate the system
    """
    def __init__(self, n_atoms, n_spatial):
        self.n_atoms = n_atoms
        self.n_spatial = n_spatial
        self.n_dofs = n_atoms * n_spatial
        # if the Hamiltonian can not be written H = T(p) + V(q), then
        # `cross_terms` is True
        self.cross_terms = False

    def H(self, snapshot):
        return self.V(snapshot) + self.kinetic_energy(snapshot)

    def V(self, snapshot):
        raise NotImplementedError("Using generic PES object")

    def __call__(self, snapshot):
        return self.V(snapshot)

    def kinetic_energy(self, snapshot):
        return 0.5*np.dot(snapshot.velocities, snapshot.momenta)

    def T(self, snapshot):
        """T = L + V; such that L = T - V; for generic action integration"""
        return self.kinetic_energy(snapshot)

    def dHdq(self, snapshot):
        dHdq = np.zeros(self.n_spatial * self.n_atoms)
        self.set_dHdq(dHdq, snapshot)
        return dHdq

    def set_dHdq(self, dHdq, snapshot):
        raise NotImplementedError("Using generic PES object")

    def dHdp(self, snapshot):
        dHdp = np.zeros(self.n_spatial * self.n_atoms)
        self.set_dHdp(dHdp, snapshot)
        return dHdp

    def set_dHdp(self, dHdp, snapshot):
        np.copyto(dHdp, snapshot.velocities)

    def d2Hdq2(self, snapshot):
        d2Hdq2 = np.zeros((self.n_dofs, self.n_dofs))
        self.set_d2Hdq2(d2Hdq2, snapshot)
        return d2Hdq2

    def set_d2Hdq2(self, d2Hdq2, snapshot):
        raise NotImplementedError("Using generic PES object")

    def d2Hdp2(self, snapshot):
        d2Hdp2 = np.zeros((self.n_dofs, self.n_dofs))
        self.set_d2Hdp2(d2Hdp2, snapshot)
        return d2Hdp2

    def set_d2Hdp2(self, d2Hdp2, snapshot):
        # TODO: this is typically constant. Shouldn't we speed it up?
        d2Hdp2.fill(0.0)
        inv_m = snapshot.topology.inverse_masses
        for i in range(len(inv_m)):
            d2Hdp2[(i,i)] = inv_m[i]

    def d2Hdqdp(self, snapshot):
        d2Hdqdp = np.zeros((self.n_dofs, self.n_dofs))
        self.set_d2Hdqdp(d2Hdqdp, snapshot)
        return d2Hdqdp

    def set_d2Hdqdp(self, d2Hdqdp, snapshot):
        return # default shouldn't even alloc these

    def d2Hdpdq(self, snapshot):
        d2Hdpdq = np.zeros((self.n_dofs, self.n_dofs))
        self.set_d2Hdpdq(d2Hdpdq, snapshot)
        return d2Hdpdq

    def set_d2Hdpdq(self, d2Hdpdq, snapshot):
        return # default shouldn't even alloc these
