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

        runnables = self.H_matrix.runnable_entries.values()
        try:
            super(MMSTHamiltonian, self).__init__(
                n_atoms=runnables[0].n_atoms,
                n_spatial=runnables[0].n_spatial
            )
        except IndexError:
            super(MMSTHamiltonian, self).__init__(0, 0)

        self.n_nuclear_dim = self.n_dofs
        self.n_dofs += self.n_electronic_states

        err_str = " not the same in all nonadiabatic matrix entries."
        for runnable in runnables:
            assert runnable.n_spatial == self.n_spatial, "n_spatial" + err_str
            assert runnable.n_atoms == self.n_atoms, "n_atoms" + err_str


    def _elect_cache(self, snap):
        """Returns a cache of the electronic parts of the Hamiltonian.

        This cache is such that `elect[(i,i)]` is $0.5(x_i^2 + p_i^2 - 1)$
        and `elect[(i,j)]` (j>i) is $x_i x_j + p_i p_j$, with $x$ and $p$
        being the MMST electronic variables.

        This is how the electronic variables appear and the MMST
        Hamiltonian, and these values are reused for the derivatives of the
        potential. Hence, a reason to cache them.
        """
        elect = {}
        for i in range(self.n_electronic_states):
            x_i = snap.electronic_coordinates[i]
            p_i = snap.electronic_momenta[i]
            elect[(i,i)] = 0.5*(x_i*x_i + p_i*p_i - 1.0)
            for j in range(i+1, self.n_electronic_states):
                x_j = snap.electronic_coordinates[j]
                p_j = snap.electronic_momenta[j]
                elect[(i,j)] = x_i*x_j + p_i*p_j
        return elect
 
    def V(self, snapshot):
        """For the MMST Hamiltonian, this is V_{eff}, the effective potential.

        We continue to define the kinetic energy as the nuclear kinetic
        energy. V_{eff} is everything else.
        """
        elect = self._elect_cache(snapshot)
        V_ij = self.H_matrix.numeric_matrix(snapshot)
       
        V = sum([elect[key] * V_ij[key] for key in self.H_matrix.keys()])
        return V

    
    def set_electronic_dHdq(self, electronic_dHdq, snapshot):
        V_ij = self.H_matrix.numeric_matrix(snapshot)

        for i in range(self.n_electronic_states):
            electronic_dHdq[i] = sum(
                [snapshot.electronic_coordinates[j] * V_ij[(i,j)]
                 for j in range(self.n_electronic_states)]
            )

    def set_dHdq(self, dHdq, snapshot):
        elect = self._elect_cache(snapshot)
        dHdq.fill(0.0)
        self._part_dHdq = np.zeros_like(dHdq)
        runnable_keys = [(i,j) 
                         for (i,j) in self.H_matrix.runnable_entries.keys() 
                         if i<=j] # upper triangular version
        # TODO: cache the nuclear dHdqs ... see use in second deriv stuff

        for key in runnable_keys:
            self.H_matrix.runnable_entries[key].set_dHdq(self._part_dHdq,
                                                         snapshot)
            np.add(self._part_dHdq * elect[key], dHdq, dHdq)


    def set_electronic_dHdp(self, electronic_dHdp, snapshot):
        V_ij = self.H_matrix.numeric_matrix(snapshot)

        for i in range(self.n_electronic_states):
            electronic_dHdp[i] = sum(
                [snapshot.electronic_momenta[j] * V_ij[(i,j)]
                 for j in range(self.n_electronic_states)]
            )

    def electronic_dHdp(self, snapshot):
        e_dHdp = np.zeros(self.n_electronic_states)
        self.set_electronic_dHdp(e_dHdp, snapshot)
        return e_dHdp

    def electronic_dHdq(self, snapshot):
        e_dHdq = np.zeros(self.n_electronic_states)
        self.set_electronic_dHdq(e_dHdq, snapshot)
        return e_dHdq

    # dHdp (for nuclear only) is still the same as standard

    def T(self, snapshot):
        """ T = L + V, such that L = T - V
        
        Since $L = P \dot{q} - H$, with $\dot{q} = dH/dp$, and $dHdp$
        includes both $p$ and $q$, this gets a little more complicated than
        the normal non-MMST implementation.
        """
        V_ij = self.H_matrix.numeric_matrix(snapshot)

        T = self.kinetic_energy(snapshot)
        for i in range(self.n_electronic_states):
            p_i = snapshot.electronic_momenta[i]
            T += V_ij[(i,i)] * p_i * p_i
            for j in range(i+1, self.n_electronic_states):
                p_j = snapshot.electronic_momenta[j]
                T += 2 * V_ij[(i,j)] * p_i * p_j

        return T

    def set_d2Hdq2(self, d2Hdq2, snapshot):
        V_ij = self.H_matrix.numeric_matrix(snapshot)
        elect = self._elect_cache(snapshot)

        # electronic-electronic block
        for i in range(self.n_electronic_states):
            for j in range(i, self.n_electronic_states):
                d2Hdq2[(i,j)] = V_ij[(i,j)]
                d2Hdq2[(j,i)] = V_ij[(i,j)]

        # this can be sped up, I think TODO: check correctness
        runnable_keys = [(i,j) 
                         for (i,j) in self.H_matrix.runnable_entries.keys() 
                         if i<=j] # upper triangular version

        nuc_d2Hdq2 = np.zeros((self.n_nuclear_dim, self.n_nuclear_dim))
        tmp = np.zeros((self.n_nuclear_dim, self.n_nuclear_dim))
        for k in runnable_keys:
            self.H_matrix.runnable_entries[k].set_d2Hdq2(tmp, snapshot)
            nuc_d2Hdq2 += tmp*elect[k]

        # nuclear-nuclear block
        for i in range(self.n_nuclear_dim):
            dof_i = self.n_electronic_states + i
            for j in range(i, self.n_nuclear_dim):
                dof_j = self.n_electronic_states + j
                d2Hdq2[(dof_i, dof_j)] = nuc_d2Hdq2[(i,j)]
                d2Hdq2[(dof_j, dof_i)] = nuc_d2Hdq2[(i,j)]

        # nuclear-electronic blocks
        nuc_dHdq = {k : self.H_matrix.runnable_entries[k].dHdq(snapshot)
                    for k in self.H_matrix.runnable_entries.keys()}
        # TODO
        for i in range(self.n_electronic_states):
            dof_i = i
            for j in range(self.n_nuclear_dim):
                dof_j = self.n_electronic_states + j
                val = sum([
                    snapshot.electronic_coordinates[k]*nuc_dHdq[(i,k)][j]
                    for k in range(self.n_electronic_states)
                ])
                d2Hdq2[(dof_i, dof_j)] = val
                d2Hdq2[(dof_j, dof_i)] = val


    def set_d2Hdp2(self, d2Hdp2, snapshot):
        V_ij = self.H_matrix.numeric_matrix(snapshot)
        elect = self._elect_cache(snapshot)

        d2Hdp2.fill(0.0)
        # nuclear-electronic blocks are zero

        # electronic-electronic
        for i in range(self.n_electronic_states):
            for j in range(i, self.n_electronic_states):
                d2Hdp2[(i,j)] = V_ij[(i,j)]
                d2Hdp2[(j,i)] = V_ij[(i,j)]
        
        # nuclear-nuclear
        for i in range(self.n_nuclear_dim):
            d2Hdp2[(i,i)] = snapshot.topology.inverse_masses[i]


    def set_d2Hdqdp(self, d2Hdqdp, snapshot):
        runnable_keys = [(i,j)
                         for (i,j) in self.H_matrix.runnable_entries.keys() 
                         if i<=j] # upper triangular version
        nuc_dHdq = {k : self.H_matrix.runnable_entries[k].dHdq(snapshot)
                    for k in self.H_matrix.runnable_entries.keys()}
        d2Hdqdp.fill(0.0)
        for i in range(self.n_electronic_states):
            dof_i = i
            for j in range(self.n_nuclear_dim):
                dof_j = self.n_electronic_states + j
                val = sum([
                    snapshot.electronic_momenta[k]*nuc_dHdq[(i,k)][j]
                    for k in range(self.n_electronic_states)
                ])
                d2Hdqdp[(dof_i, dof_j)] = val
                d2Hdqdp[(dof_j, dof_i)] = val


    def set_d2Hdpdq(self, d2Hdpdq, snapshot):
        self.set_d2Hdqdp(d2Hdpdq, snapshot)
        np.transpose(d2Hdpdq)

