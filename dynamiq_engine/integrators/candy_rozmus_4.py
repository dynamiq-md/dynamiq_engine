from integrator import Integrator
import numpy as np

import math

class CandyRozmus4(Integrator):
    """Fourth-order integrator by Candy and Rozmus.

    References
    ----------
    """
    def __init__(self, dt, potential, n_frames=1):
        super(CandyRozmus4, self).__init__(dt)
        self._a_k = [
            0.5*(1.0 - 1.0/math.sqrt(3.0))*self.dt,
            1.0/sqrt(3.0)*self.dt,
            -0.5*(1.0 - 1.0/math.sqrt(3.0))*self.dt,
            0.5*(1.0 + 1.0/math.sqrt(3.0))*self.dt,
        ]
        self._b_k = [
            0.0,
            0.5*(0.5 + 1.0/math.sqrt(3.0))*self.dt,
            0.5 * self.dt,
            0.5*(0.5 - 1.0/math.sqrt(3.0))*self.dt
        ]
        n_spatial = potential.n_spatial
        n_atoms = potential.n_atoms
        self.local_dHdq = np.zeros((self.n_spatial * self.n_atoms))
        self.local_dHdp = np.zeros((self.n_spatial * self.n_atoms))

    def momentum_update(self, potential, snap, k):
        potential.set_dHdq(self.local_dHdq, snap)
        self.local_dHdq *= self._a_k[k]
        np.add(snap.momenta, self.local_dHdq, snap.momenta)

    def position_update(self, potential, snap, k):
        potential.set_dHdp(self.local_dHdp, snap)
        self.local_dHdp *= self._b_k[k]
        np.subtract(snap.coordinates, self.local_dHdp, snap.coordinates)

    def action_step(self, potential, old_snap, new_snap, k):
        pass

    def step(self, potential, old_snap, new_snap):
        intermediate.copy_from(old_snap)
        for k in range(4):
            momentum_step(potential, intermediate, k)
            # TODO: monodromy and action
            position_step(potential, intermediate, k)

        # wrap PBCs if necessary
        new_snap.copy_from(intermediate)


# to be done later
class CandyRozmus4Monodromy(CandyRozmus4):
    """Fourth-order integrator by Candy and Rozmus, including monodromy
    propagation.

    References
    ----------
    """
    pass

class GeneralizedCandyRozmus4Monodromy(CandyRozmus4Monodromy):
    """Candy-Rozmus integrator including monodromy treatment when
    Hamiltonian is not separable.
    """
    pass
