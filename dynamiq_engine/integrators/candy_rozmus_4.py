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
            1.0/math.sqrt(3.0)*self.dt,
            -1.0/math.sqrt(3.0)*self.dt,
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
        self.local_dHdq = np.zeros(n_spatial * n_atoms)
        self.local_dHdp = np.zeros(n_spatial * n_atoms)

    def prepare(self, feature_list):
        import dynamiq_engine.features as dynq_f
        import openpathsampling.engines.features as paths_f
        self.feature_list = feature_list
        # TODO: get another integrator to support electronic dofs; I don't
        # think this one tehcnically should
        supported_features = [
            paths_f.coordinates,
            paths_f.xyz,
            paths_f.topology,
            dynq_f.momenta,
            dynq_f.velocities,
            dynq_f.action,
            dynq_f.electronic_coordinates,
            dynq_f.electronic_momenta
            # TODO: add support for monodromy or for prefactor
        ]
        #print self.feature_list
        #print supported_features
        for f in self.feature_list:
            if f not in supported_features:
                raise RuntimeError("Feature " + str(f) +
                                   " not supported by this integrator")

        # Two potential approaches: either we build the `update` function
        # here, or we have a bunch of if-statements in the main `step`
        # function. Either way, we have the option here of overriding `step`
        # with a customized, faster version based on the `feature_list`.

        pre_step = []
        pre_momentum = []
        post_momentum = []
        pre_position = []
        post_position = []
        post_step = []
        momentum = [self.momentum_calculate, self.momentum_update]
        position = [self.position_calculate, self.position_update]

        if dynq_f.electronic_momenta in self.feature_list:
            pre_momentum.append(self.electronic_momentum_calculate)
            post_momentum.append(self.electronic_momentum_update)

        if dynq_f.electronic_coordinates in self.feature_list:
            pre_position.append(self.electronic_position_calculate)
            post_position.append(self.electronic_position_update)

        if dynq_f.action in self.feature_list:
            post_momentum.append(self.action_update)
            self.local_S = 0.0

        self.update_steps = (pre_step +
                             pre_momentum + momentum + post_momentum +
                             pre_position + position + post_position +
                             post_step)

    def reset(self):
        import dynamiq_engine.features as dynq_f
        # import openpathsampling.features as paths_f

        if dynq_f.action in self.feature_list:
            self.local_S = 0.0
        pass

    def momentum_calculate(self, potential, snap, k):
        potential.set_dHdq(self.local_dHdq, snap)
        self.local_dHdq *= self._b_k[k]

    def momentum_update(self, potential, snap, k):
        np.subtract(snap.momenta, self.local_dHdq, snap.momenta)

    def electronic_momentum_calculate(self, potential, snap, k):
        potential.set_electronic_dHdq(self.local_electronic_dHdq, snap)
        self.local_electronic_dHdq *= self._b_k[k]

    def electronic_momentum_update(self, potential, snap, k):
        np.subtract(snap.electronic_momenta, self.local_electronic_dHdq,
                    snap.electronic_momenta)

    def position_calculate(self, potential, snap, k):
        potential.set_dHdp(self.local_dHdp, snap)
        self.local_dHdp *= self._a_k[k]

    def position_update(self, potential, snap, k):
        np.add(snap.coordinates, self.local_dHdp, snap.coordinates)

    def electronic_position_calculate(self, potential, snap, k):
        potential.set_electronic_dHdp(self.local_electronic_dHdp, snap)
        self.local_electronic_dHdp *= self._a_k[k]

    def electronic_position_update(self, potential, snap, k):
        np.add(snap.electronic_coordinates, self.local_electronic_dHdp,
               snap.electronic_coordinates)

    def action_update(self, potential, snap, k):
        self.local_S += self._a_k[k]*potential.T(snap)
        self.local_S -= self._b_k[k]*potential.V(snap)
        snap.action = self.local_S

    def step(self, potential, old_snap=None, new_snap=None):
        old_snap.copy_to(new_snap)
        #new_snap.copy_from(old_snap)
        for k in range(4):
            for update in self.update_steps:
                update(potential, new_snap, k)
        # wrap PBCs if necessary


class CandyRozmus4MMST(CandyRozmus4):
    def __init__(self, dt, potential, n_frames=1):
        super(CandyRozmus4MMST, self).__init__(dt, potential, n_frames)
        self.local_electronic_dHdp = np.zeros(potential.n_electronic_states)
        self.local_electronic_dHdq = np.zeros(potential.n_electronic_states)
        import dynamiq_engine.features as dynq_f
        import openpathsampling.engines.features as paths_f
        self.prepare([paths_f.coordinates, dynq_f.momenta,
                      dynq_f.electronic_coordinates,
                      dynq_f.electronic_momenta])


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
