import dynamiq_engine as dynq
import numpy as np
from tools import *

from dynamiq_engine.integrators.monodromy import *

import dynamiq_engine.potentials as pes
from example_systems import ho_2_1, tully
import openpathsampling.engines.features as paths_f
import openpathsampling.engines as peng
import dynamiq_engine.features as dynq_f

MonodromySnapshot = peng.SnapshotFactory(
    name="MonodromySnapshot",
    features=[paths_f.coordinates, dynq_f.momenta, dynq_f.monodromy,
              paths_f.topology]
)

class testStandardMonodromy(object):
    def setup(self):
        self.monodromy = StandardMonodromy()
        self.potential = ho_2_1.potential
        self.topology = ho_2_1.topology
        self.integ = ho_2_1.integrator

        self.monodromy.prepare(self.integ)

        self.snap0 = MonodromySnapshot(
            coordinates=np.array([1.0]),
            momenta=np.array([1.0]),
            topology=self.topology
        )

    def test_prepare(self):
        mono = StandardMonodromy()
        mono.prepare(self.integ)
        assert_equal(mono.second_derivatives.cross_terms, False)
        assert_equal(mono.second_derivatives, self.potential)
        for matrix in [mono._local_dMqq_dt, mono._local_dMqp_dt,
                       mono._local_dMpq_dt, mono._local_dMpp_dt,
                       mono._local_Hpp, mono._local_Hqq]:
            assert_equal(matrix, np.array([[0.0]]))
        for matrix in [mono._local_Hpq, mono._local_Hqp]:
            assert_equal(matrix, None)

    def test_reset(self):
        fresh_snap = MonodromySnapshot(
            coordinates=np.array([1.0]),
            momenta=np.array([1.0]),
            topology=self.topology
        )
        # check that the snapshot has the monodromy feature, although the
        # matrix itself is unset
        assert_equal(fresh_snap.Mqq, None)
        assert_equal(fresh_snap.Mqp, None)
        assert_equal(fresh_snap.Mpq, None)
        assert_equal(fresh_snap.Mpp, None)
        # check that reset sets it correctly
        self.monodromy.reset(fresh_snap)
        assert_equal(fresh_snap.Mqq, np.array([[1.0]]))
        assert_equal(fresh_snap.Mpp, np.array([[1.0]]))
        assert_equal(fresh_snap.Mqp, np.array([[0.0]]))
        assert_equal(fresh_snap.Mpq, np.array([[0.0]]))
        # TODO: change it, check that resetting goes back
        raise SkipTest

    def test_dMqq_dt(self):
        self.integ.prepare([paths_f.coordinates, dynq_f.momenta])
        self.monodromy.prepare(self.integ)
        self.integ.reset(self.snap0)
        #dMqq_dt = self.monodromy.dMqq_dt(self.potential, self.snap0)
        raise SkipTest

    def test_dMqp_dt(self):
        raise SkipTest

    def test_dMpq_dt(self):
        raise SkipTest

    def test_dMpp_dt(self):
        raise SkipTest


class testStandardMonodromyMMST(object):
    def setup(self):
        self.monodromy = StandardMonodromy()
        self.potential = tully.potential
        self.topology = tully.topology
        self.integ = tully.integrator

    def test_prepare(self):
        self.monodromy.prepare(self.integ)
        assert_equal(self.monodromy.second_derivatives.cross_terms, True)
        assert_equal(self.monodromy.second_derivatives, self.potential)
        mono = self.monodromy  # to simplify names
        for matrix in [mono._local_dMqq_dt, mono._local_dMqp_dt,
                       mono._local_dMpq_dt, mono._local_dMpp_dt,
                       mono._local_Hpp, mono._local_Hqq,
                       mono._local_Hqp, mono._local_Hpq, mono._tmp]:
            assert_array_almost_equal(matrix, np.array([[0.0, 0.0, 0.0],
                                                        [0.0, 0.0, 0.0],
                                                        [0.0, 0.0, 0.0]]))
