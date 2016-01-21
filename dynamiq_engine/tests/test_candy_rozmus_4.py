import dynamiq_engine as dynq
import numpy as np
from tools import *

from dynamiq_engine.integrators.candy_rozmus_4 import *

class testCandyRozmus4(object):
    def setup(self):
        ho = dynq.potentials.interactions.HarmonicOscillatorInteraction(
            k=2.0, x0=1.0
        )
        self.potential = dynq.potentials.OneDimensionalInteractionModel(ho)
        self.topology = dynq.Topology(masses=np.array([0.5]), 
                                      potential=self.potential)
        self.integ = CandyRozmus4(0.01, self.potential)
        self.snap0 = dynq.Snapshot(
            coordinates=np.array([1.0]),
            momenta=np.array([0.0]),
            topology=self.topology
        )

    def test_cr4_step(self):
        new_snapshot = dynq.Snapshot(coordinates=np.array([0.0]),
                                     momenta=np.array([0.0]),
                                     topology=self.topology)
        # snap0 is at he minimum; it shouldn't move
        self.integ.step(self.potential, self.snap0, new_snapshot)
        assert_array_almost_equal(self.snap0.coordinates,
                                  new_snapshot.coordinates)
        assert_array_almost_equal(self.snap0.momenta,
                                  new_snapshot.momenta)
        for i in range(10):
            self.integ.step(self.potential, new_snapshot, new_snapshot)
        assert_array_almost_equal(self.snap0.coordinates,
                                  new_snapshot.coordinates)
        assert_array_almost_equal(self.snap0.momenta,
                                  new_snapshot.momenta)

        snap1 = self.snap0.copy()
        snap1.coordinates=np.array([0.0])
        self.integ.step(self.potential, snap1, snap1)
        # TODO: just need the asserts here
        print snap1.coordinates, snap1.momenta
        #assert_array_almost_equal(snap1.coordinates,
                                  #np.array([1.0-np.cos(2*0.01)]))
        #assert_array_almost_equal(snap1.momenta,
                                  #np.array([-np.sin(2*0.01)]))
        
        raise SkipTest
