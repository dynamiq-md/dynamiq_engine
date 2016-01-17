import dynamiq_engine as dynq
from tools import *

from dynamiq_engine.potentials.pairwise_interactions import *

class testHarmonicOscillatorInteraction(object):
    def setup(self):
        self.ho = HarmonicOscillatorInteraction(k=2.0, x0=1.0)

    def test_f(self):
        tests = {
            0.0 : 1.0, # = 0.5*2.0*(0.0-1.0)^2
            1.0 : 0.0,
            0.5 : 0.25, # = 0.5*2.0*(0.5-1.0)^2
            2.0 : 1.0
        }
        for val in tests.keys():
            assert_almost_equal(self.ho.f(val), tests[val])
            assert_almost_equal(self.ho(val), tests[val])

    def test_dfdx(self):
        tests = {
            0.0 : -2.0,
            1.0 : 0.0,
            0.5 : -1.0,
            2.0 : 2.0
        }
        for val in tests.keys():
            assert_almost_equal(self.ho.dfdx(val), tests[val])

    def test_d2fdx2(self):
        tests = {
            0.0 : 2.0,
            1.0 : 2.0,
            0.5 : 2.0,
            2.0 : 2.0
        }
        for val in tests.keys():
            assert_almost_equal(self.ho.d2fdx2(val), tests[val])

    
