import dynamiq_engine as dynq
import numpy as np
from tools import *
import dynamiq_engine.potentials as pes

from dynamiq_engine.potentials.potential_energy_surface import *

class testOneDimensionalInteractionModel(object):
    def setup(self):
        self.ho = pes.interactions.HarmonicOscillatorInteraction(k=2.0, x0=1.0)
        self.ho_pot = OneDimensionalInteractionModel(self.ho)
        ho_topol = dynq.Topology(masses=np.array([0.5]), potential=self.ho_pot)
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
        raise SkipTest

    def test_d2Hdpdq(self):
        raise SkipTest

    def test_d2Hdqdp(self):
        raise SkipTest

    def test_d2Hdp2(self):
        raise SkipTest
