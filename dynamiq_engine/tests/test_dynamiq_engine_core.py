import dynamiq_engine as dynq
import numpy as np
from tools import *
import stubs
from dynamiq_engine.dynamiq_engine_core import *

class testTopology(object):
    def setup(self):
        pes = stubs.PotentialStub()
        masses = np.array([2.0])
        self.topology = Topology(masses, pes)

    def test_inverse_masses(self):
        assert_array_almost_equal(self.topology.inverse_masses, 
                                  np.array([0.5]))


class testSnapshot(object):
    def setup(self):
        self.topology = Topology(
            masses=np.array([2.0]),
            potential=stubs.PotentialStub()
        )
        self.snap = Snapshot(
            coordinates=np.array([1.0]),
            momenta=np.array([1.0]),
            topology=self.topology
        )

    def test_velocities(self):
        assert_array_almost_equal(self.snap.velocities, np.array([0.5]))

    def test_xyz(self):
        assert_array_almost_equal(self.snap.coordinates, self.snap.xyz)


class testDynamiqEngine(object):
    def setup(self):
        from example_systems import ho_2_1
        self.topology = ho_2_1.topology
        self.potential = ho_2_1.potential
        self.integrator = ho_2_1.integrator
        self.snap = Snapshot(
            coordinates=np.array([0.0]),
            momenta=np.array([0.0]),
            topology=self.topology
        )
        self.engine = DynamiqEngine(self.potential, 
                                    self.integrator,
                                    self.snap)

    def test_current_snapshot(self):
        snap = Snapshot(
            coordinates=np.array([1.0]),
            momenta=np.array([1.0]),
            topology=self.topology
        )
        self.engine.current_snapshot = snap
        loaded = self.engine.current_snapshot

        assert_array_almost_equal(snap.coordinates, loaded.coordinates)
        assert_array_almost_equal(snap.momenta, loaded.momenta)


    def test_generate_next_frame(self):
        snap = Snapshot(
            coordinates=np.array([1.0]),
            momenta=np.array([1.0]),
            topology=self.topology
        )
        self.engine.current_snapshot = snap
        self.engine.start()
        newsnap = self.engine.generate_next_frame()
        raise SkipTest
