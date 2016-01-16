import dynamiq_engine as dynq
from tools import *

from dynamiq_engine.potentials.pairwise_interactions import *

class testHarmonicOscillatorInteraction(object):
    def setup(self):
        self.ho = HarmonicOscillatorInteraction(k=2.0, x0=1.0)

    def test_f(self):
        raise SkipTest

    def test_dfdx(self):
        raise SkipTest

    def test_d2fdx2(self):
        raise SkipTest

    
