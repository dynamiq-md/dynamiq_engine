import dynamiq_engine as dynq
import numpy as np
from tools import *

from dynamiq_engine.integrators.monodromy import *

import dynamiq_engine.potentials as pes
from example_systems import ho_2_1, tully
import openpathsampling.features as paths_f
import dynamiq_engine.features as dynq_f

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
        assert_equal(self.monodromy.second_derivatives.cross_terms, False)
        assert_equal(self.monodromy.second_derivatives, self.potential)
        mono = self.monodromy  # to simplify names
        for matrix in [mono._local_dMqq_dt, mono._local_dMqp_dt,
                       mono._local_dMpq_dt, mono._local_dMpp_dt,
                       mono._local_Hpp, mono._local_Hqq]:
            assert_equal(matrix, np.array([[0.0]]))
        for matrix in [mono._local_Hpq, mono._local_Hqp]:
            assert_equal(matrix, None)

    def test_dMqq_dt(self):
        self.integ.prepare([paths_f.coordinates, dynq_f.momenta])
        self.monodromy.prepare(self.integ)
        self.integ.reset(self.snap0)
        #dMqq_dt = self.monodromy.dMqq_dt(self.potential, self.snap0)


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
