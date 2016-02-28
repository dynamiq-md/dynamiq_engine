import dynamiq_engine as dynq
from tools import *

from dynamiq_engine.potentials.pairwise_interactions import *
from example_systems import ho_2_1, anharmonic_morse

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

    def test_dfdx(self):
        tests = { 0.0 : 0.0, 1.0 : 0.0, 0.5 : 0.0 }
        check_function(self.cst.dfdx, tests)

    def test_d2fdx2(self):
        tests = { 0.0 : 0.0, 1.0 : 0.0, 0.5 : 0.0 }
        check_function(self.cst.d2fdx2, tests)

class testHarmonicOscillatorInteraction(object):
    def setup(self):
        self.ho = ho_2_1.potential
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

    def test_f(self):
        tests = {
            0.0 : 1.0, # = 0.5*2.0*(0.0-1.0)^2
            1.0 : 0.0,
            0.5 : 0.25, # = 0.5*2.0*(0.5-1.0)^2
            2.0 : 1.0
        }
        check_function(self.ho.f, tests)

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

    def test_H(self):
        assert_almost_equal(self.ho.H(self.test_snap1), 1.0)
        assert_almost_equal(self.ho.H(self.test_snap2), 4.25)
        assert_almost_equal(self.ho.H(self.test_snap3), 1.25)

    def test_T(self):
        # Definition of T is $L + V = p * dH/dp - H + V$; test this directly.
        # Use a lambda rather than nested def because nested def screws with
        # my in-editor test runner
        explicit_T = lambda pes, snap : (
            np.dot(snap.momenta, pes.dHdp(snap))
            - self.ho.H(snap) + self.ho.V(snap)
        )
        assert_almost_equal(self.ho.T(self.test_snap1),
                            explicit_T(self.ho, self.test_snap1))
        assert_almost_equal(self.ho.T(self.test_snap2),
                            explicit_T(self.ho, self.test_snap2))
        assert_almost_equal(self.ho.T(self.test_snap3),
                            explicit_T(self.ho, self.test_snap3))

    def test_dHdq(self):
        assert_array_almost_equal(self.ho.dHdq(self.test_snap1), 
                                  np.array([0.0]))
        assert_array_almost_equal(self.ho.dHdq(self.test_snap2), 
                                  np.array([-1.0]))
        assert_array_almost_equal(self.ho.dHdq(self.test_snap3), 
                                  np.array([2.0]))

    def test_dHdp(self):
        assert_array_almost_equal(self.ho.dHdp(self.test_snap1),
                                  np.array([2.0]))
        assert_array_almost_equal(self.ho.dHdp(self.test_snap2),
                                  np.array([4.0]))
        assert_array_almost_equal(self.ho.dHdp(self.test_snap3),
                                  np.array([-1.0]))

    def test_d2Hdq2(self):
        raise SkipTest

    def test_d2Hdpdq(self):
        raise SkipTest

    def test_d2Hdqdp(self):
        raise SkipTest

    def test_d2Hdp2(self):
        raise SkipTest


class testTanhInteraction(object):
    def setup(self):
        self.tanh = TanhInteraction(a=0.75, V0=0.1, R0=0.5)

    def test_f(self):
        tests = {
            0.0 : -0.0358357398350786,
            0.5 : 0.0,
            1.0 : 0.0358357398350786,
            2.0 : 0.0809301070201781
        }
        check_function(self.tanh.f, tests)

    def test_dfdx(self):
        tests = {
            0.0 : 0.0653684981285442,
            0.5 : 0.0750000000000000,
            1.0 : 0.0653684981285442,
            2.0 : 0.0258773833327689
        }
        check_function(self.tanh.dfdx, tests)

    def test_d2fdx2(self):
        tests = {
            0.0 : 0.0351379273851650,
            0.5 : 0.0,
            1.0 : -0.0351379273851650,
            2.0 : -0.0314138910378474
        }
        check_function(self.tanh.d2fdx2, tests)


class testMorseInteraction(object):
    def setup(self):
        self.morse = MorseInteraction(D=30.0, beta=0.08, x0=0.5)

    def test_f(self):
        tests = {
            -1.0 : 0.487663414879601,
            0.0 : 0.0499655787054630,
            0.5 : 0.0,
            5.0 : 2.74198811453729
        }
        check_function(self.morse.f, tests)

    def test_dfdx(self):
        tests = {
            -1.0 : -0.690011033961740,
            0.0 : -0.203886208716337,
            0.5 : 0.0,
            5.0 : 1.01243553653309
        }
        check_function(self.morse.dfdx, tests)

    def testd2fdx2(self):
        tests = {
            -1.0 : 0.543360556440359,
            0.0 : 0.432293130684491,
            0.5 : 0.384000000000000,
            5.0 : 0.105918023365982
        }
        check_function(self.morse.d2fdx2, tests)

class testQuarticInteraction(object):
    def setup(self):
        self.quartic = QuarticInteraction(1.5, 1.25, 2.0, 1.0, 0.25, 0.5)

    def test_f(self):
        tests = {
            -1.0 : 6.625,
            0.0 : 0.1875,
            0.5 : 0.25,
            2.0 : 18.0625
        }
        check_function(self.quartic.f, tests)

    def test_dfdx(self):
        tests = {
            -1.0 : -16.8125,
            0.0 : -0.8125,
            0.5 : 1.0,
            2.0 : 35.6875
        }
        check_function(self.quartic.dfdx, tests)

    def test_d2fdx2(self):
        tests = {
            -1.0 : 33.25,
            0.0 : 4.75,
            0.5 : 4.0,
            2.0 : 55.75
        }
        check_function(self.quartic.d2fdx2, tests)

class testGaussianInteraction(object):
    def setup(self):
        self.gaussian = GaussianInteraction(A=2.0, alpha=0.25, x0=0.5)

    def test_f(self):
        tests = {
            0.0 : 1.87882612562695,
            0.5 : 2.0,
            1.0 : 1.87882612562695,
            2.0 : 1.13956564946185
        }
        check_function(self.gaussian.f, tests)

    def test_dfdx(self):
        tests = {
            0.0 : 0.469706531406738,
            0.5 : 0.0,
            1.0 : -0.469706531406738,
            2.0 : -0.854674237096384
        }
        check_function(self.gaussian.dfdx, tests)

    def test_d2fdx2(self):
        tests = {
            0.0 : -0.821986429961791,
            0.5 : -1.0,
            1.0 : -0.821986429961791,
            2.0 : 0.0712228530913653
        }
        check_function(self.gaussian.d2fdx2, tests)

