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
        topology = dynq.Topology(masses=np.array([0.5]), 
                                 potential=self.potential)
        self.snap0 = dynq.Snapshot(
            coordinates=np.array([1.0]),
            momenta=np.array([0.0])
        )
        pass

    def test_cr4_step(self):
        raise SkipTest
