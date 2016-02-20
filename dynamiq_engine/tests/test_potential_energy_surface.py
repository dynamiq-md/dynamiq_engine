import dynamiq_engine as dynq
import numpy as np
from tools import *
import dynamiq_engine.potentials as pes

from dynamiq_engine.potentials.potential_energy_surface import *

class testOneDimensionalInteractionModel(object):
    def setup(self):
        from example_systems import ho_2_1
        self.pot = ho_2_1.potential
        topology = ho_2_1.topology
        self.test_snap1 = dynq.Snapshot(
            coordinates=np.array([1.0]),
            momenta=np.array([1.0]),
            topology=topology
        )
        self.test_snap2 = dynq.Snapshot(
            coordinates=np.array([0.5]),
            momenta=np.array([2.0]),
            topology=topology
        )
        self.test_snap3 = dynq.Snapshot(
            coordinates=np.array([2.0]),
            momenta=np.array([-0.5]),
            topology=topology
        )

    def test_H(self):
        assert_almost_equal(self.pot.H(self.test_snap1), 1.0)
        assert_almost_equal(self.pot.H(self.test_snap2), 4.25)
        assert_almost_equal(self.pot.H(self.test_snap3), 1.25)

    def test_T(self):
        # Definition of T is $L + V = p * dH/dp - H + V$; test this directly.
        # Use a lambda rather than nested def because nested def screws with
        # my in-editor test runner
        explicit_T = lambda pes, snap : (
            np.dot(snap.momenta, pes.dHdp(snap))
            - self.pot.H(snap) + self.pot.V(snap)
        )

        assert_almost_equal(self.pot.T(self.test_snap1),
                            explicit_T(self.pot, self.test_snap1))
        assert_almost_equal(self.pot.T(self.test_snap2),
                            explicit_T(self.pot, self.test_snap2))
        assert_almost_equal(self.pot.T(self.test_snap3),
                            explicit_T(self.pot, self.test_snap3))


    def test_dHdq(self):
        assert_array_almost_equal(self.pot.dHdq(self.test_snap1), 
                                  np.array([0.0]))
        assert_array_almost_equal(self.pot.dHdq(self.test_snap2), 
                                  np.array([-1.0]))
        assert_array_almost_equal(self.pot.dHdq(self.test_snap3), 
                                  np.array([2.0]))

    def test_dHdp(self):
        assert_array_almost_equal(self.pot.dHdp(self.test_snap1),
                                  np.array([2.0]))
        assert_array_almost_equal(self.pot.dHdp(self.test_snap2),
                                  np.array([4.0]))
        assert_array_almost_equal(self.pot.dHdp(self.test_snap3),
                                  np.array([-1.0]))


    def test_d2Hdq2(self):
        raise SkipTest

    def test_d2Hdpdq(self):
        raise SkipTest

    def test_d2Hdqdp(self):
        raise SkipTest

    def test_d2Hdp2(self):
        raise SkipTest
