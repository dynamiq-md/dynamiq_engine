import dynamiq_engine as dynq
from tools import *

from dynamiq_engine.potentials.pairwise_interactions import *

class testConstantInteraction(object):
    def setup(self):
        self.cst = ConstantInteraction(value=3.5)

    def test_f(self):
        tests = {
            0.0 : 3.5,
            1.0 : 3.5,
            0.5 : 3.5
        }
        check_function(self.cst.f, tests)
        check_function(self.cst, tests)

    def test_dfdx(self):
        tests = { 0.0 : 0.0, 1.0 : 0.0, 0.5 : 0.0 }
        check_function(self.cst.dfdx, tests)

    def test_d2fdx2(self):
        tests = { 0.0 : 0.0, 1.0 : 0.0, 0.5 : 0.0 }
        check_function(self.cst.d2fdx2, tests)

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
        check_function(self.ho.f, tests)
        check_function(self.ho, tests)

    def test_dfdx(self):
        tests = {
            0.0 : -2.0,
            1.0 : 0.0,
            0.5 : -1.0,
            2.0 : 2.0
        }
        check_function(self.ho.dfdx, tests)

    def test_d2fdx2(self):
        tests = {
            0.0 : 2.0,
            1.0 : 2.0,
            0.5 : 2.0,
            2.0 : 2.0
        }
        check_function(self.ho.d2fdx2, tests)

    
