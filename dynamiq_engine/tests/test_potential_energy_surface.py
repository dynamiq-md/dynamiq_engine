import dynamiq_engine as dynq
import numpy as np
from tools import *
import dynamiq_engine.potentials as pes

from dynamiq_engine.potentials.potential_energy_surface import *

class testOneDimensionalInteractionModel(object):
    def setup(self):
        from example_systems import ho_2_1
        self.ho_pot = ho_2_1.potential
        ho_topol = ho_2_1.topology
        self.ho_test_snaps = [
            dynq.Snapshot(coordinates=np.array([1.0]),
                          momenta=np.array([1.0]),
                          topology=ho_topol),
            dynq.Snapshot(coordinates=np.array([0.5]),
                          momenta=np.array([2.0]),
                          topology=ho_topol),
            dynq.Snapshot(coordinates=np.array([2.0]),
                          momenta=np.array([-0.5]),
                          topology=ho_topol)
        ]

        self.morse = pes.interactions.MorseInteraction(30.0, 0.08, 0.5)
        self.morse_pot = OneDimensionalInteractionModel(self.morse)
        morse_topol = dynq.Topology(masses=np.array([0.2]),
                                    potential=self.morse_pot)
        self.morse_test_snaps = [
            dynq.Snapshot(coordinates=np.array([0.0]),
                          momenta=np.array([0.4]),
                          topology=morse_topol),
            dynq.Snapshot(coordinates=np.array([-1.0]),
                          momenta=np.array([0.6]),
                          topology=morse_topol),
            dynq.Snapshot(coordinates=np.array([5.0]),
                          momenta=np.array([0.8]),
                          topology=morse_topol)
        ]


    def test_H(self):
        tests = {
            self.ho_test_snaps[0] : 1.0,
            self.ho_test_snaps[1] : 4.25,
            self.ho_test_snaps[2] : 1.25
        }
        check_function(self.ho_pot.H, tests)

    def test_T(self):
        # Definition of T is $L + V = p * dH/dp - H + V$; test this directly.
        # Use a lambda rather than nested def because nested def screws with
        # my in-editor test runner
        explicit_T = lambda pes, snap : (
            np.dot(snap.momenta, pes.dHdp(snap)) - pes.H(snap) + pes.V(snap)
        )
        tests = {s : explicit_T(self.ho_pot, s) for s in self.ho_test_snaps}
        check_function(self.ho_pot.T, tests)

    def test_dHdq(self):
        tests = {
            self.ho_test_snaps[0] : np.array([0.0]),
            self.ho_test_snaps[1] : np.array([-1.0]),
            self.ho_test_snaps[2] : np.array([2.0])
        }
        check_function(self.ho_pot.dHdq, tests)

    def test_dHdp(self):
        tests = {
            self.ho_test_snaps[0] : np.array([2.0]),
            self.ho_test_snaps[1] : np.array([4.0]),
            self.ho_test_snaps[2] : np.array([-1.0])
        }
        check_function(self.ho_pot.dHdp, tests)

    def test_d2Hdq2(self):
        # HO tests
        tests = {s : np.array([[2.0]]) for s in self.ho_test_snaps}
        check_function(self.ho_pot.d2Hdq2, tests)

        # Morse tests
        # see also test_pairwise_interaction, where these are used
        tests = {
            self.morse_test_snaps[0] : 0.432293130684491,
            self.morse_test_snaps[1] : 0.543360556440359,
            self.morse_test_snaps[2] : 0.105918023365982
        }
        check_function(self.morse_pot.d2Hdq2, tests)


    def test_d2Hdpdq(self):
        tests = {s : np.array([[0.0]]) for s in self.ho_test_snaps}
        check_function(self.ho_pot.d2Hdpdq, tests)

        local_d2Hdpdq = None
        for s in self.ho_test_snaps:
            self.ho_pot.set_d2Hdpdq(local_d2Hdpdq, s)

    def test_d2Hdqdp(self):
        tests = {s : np.array([[0.0]]) for s in self.ho_test_snaps}
        check_function(self.ho_pot.d2Hdqdp, tests)

        local_d2Hdqdp = None
        for s in self.ho_test_snaps:
            self.ho_pot.set_d2Hdqdp(local_d2Hdqdp, s)

    def test_d2Hdp2(self):
        tests = {s : np.array([[2.0]]) for s in self.ho_test_snaps}
        check_function(self.ho_pot.d2Hdp2, tests)

        # Morse tests
        tests = {s : np.array([[5.0]]) for s in self.morse_test_snaps}
        check_function(self.morse_pot.d2Hdp2, tests)

