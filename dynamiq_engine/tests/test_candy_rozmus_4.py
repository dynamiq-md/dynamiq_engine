import dynamiq_engine as dynq
import numpy as np
from tools import *

from dynamiq_engine.integrators.candy_rozmus_4 import *

class testCandyRozmus4(object):
    def setup(self):
        self.ho = dynq.potentials.interactions.HarmonicOscillatorInteraction(
            k=2.0, x0=1.0
        )
        self.potential = dynq.potentials.OneDimensionalInteractionModel(self.ho)
        self.topology = dynq.Topology(masses=np.array([0.5]), 
                                      potential=self.potential)
        self.integ = CandyRozmus4(0.01, self.potential)
        self.snap0 = dynq.Snapshot(
            coordinates=np.array([1.0]),
            momenta=np.array([0.0]),
            topology=self.topology
        )

    def exact_ho(self, initial_snap, time):
        m = initial_snap.topology.masses[0]
        omega = np.sqrt(self.ho.k / m)
        x0 = self.ho.x0
        p0 = initial_snap.momenta[0]
        q0 = initial_snap.coordinates[0] - x0
        cos_wt = np.cos(omega*time)
        sin_wt = np.sin(omega*time)
        state_at_t = {
            'q' : np.array([q0*cos_wt + p0/m/omega*sin_wt + x0]),
            'p' : np.array([p0*cos_wt - q0*m*omega*sin_wt])
        }
        return state_at_t

    def test_cr4_step(self):
        new_snap = dynq.Snapshot(coordinates=np.array([0.0]),
                                 momenta=np.array([0.0]),
                                 topology=self.topology)
        # snap0 is at he minimum; it shouldn't move
        self.integ.step(self.potential, self.snap0, new_snap)
        assert_array_almost_equal(self.snap0.coordinates, new_snap.coordinates)
        assert_array_almost_equal(self.snap0.momenta, new_snap.momenta)
        for i in range(10):
            self.integ.step(self.potential, new_snap, new_snap)
        assert_array_almost_equal(self.snap0.coordinates, new_snap.coordinates)
        assert_array_almost_equal(self.snap0.momenta, new_snap.momenta)

        snap1 = dynq.Snapshot(coordinates=np.array([0.0]),
                              momenta=np.array([0.0]),
                              topology=self.topology)
        self.integ.step(self.potential, snap1, new_snap)
        exact_0x01 = self.exact_ho(snap1, 0.01)
        assert_array_almost_equal(new_snap.coordinates, exact_0x01['q'])
        assert_array_almost_equal(new_snap.momenta, exact_0x01['p'])

        for i in range(9):
            self.integ.step(self.potential, new_snap, new_snap)
        exact_0x10 = self.exact_ho(snap1, 0.10)
        assert_array_almost_equal(new_snap.coordinates, exact_0x10['q'])
        assert_array_almost_equal(new_snap.momenta, exact_0x10['p'])
