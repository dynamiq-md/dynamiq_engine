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
        pass

    def velocities(self):
        raise SkipTest

    def xyz(self):
        raise SkipTest


class testDynamiqEngine(object):
    def setup(self):
        pass

