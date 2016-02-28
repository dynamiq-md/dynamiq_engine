import dynamiq_engine as dynq
import numpy as np
from tools import *

from dynamiq_engine.integrators.monodromy import *

import dynamiq_engine.potentials as pes
from example_systems import ho_2_1, tully

class testStandardMonodromy(object):
    def setup(self):
        self.monodromy = StandardMonodromy()
        self.potential = ho_2_1.potential
        self.topology = ho_2_1.topology
        self.integ = ho_2_1.integrator

        self.snap0 = dynq.Snapshot(
            coordinates=np.array([1.0]),
            momenta=np.array([1.0]),
            topology=self.topology
        )

    def test_prepare(self):
        self.monodromy.prepare(self.integ)
        raise SkipTest


class testStandardMonodromyMMST(object):
    def setup(self):
        self.monodromy = StandardMonodromy()
        self.potential = tully.potential
        self.topology = tully.topology
        self.integ = tully.integrator

    def test_prepare(self):
        self.monodromy.prepare(self.integ)
        raise SkipTest

