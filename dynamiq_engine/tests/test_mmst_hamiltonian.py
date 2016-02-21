import dynamiq_engine as dynq
import dynamiq_engine.potentials as pes
import numpy as np
from tools import *

from dynamiq_engine.potentials.mmst_hamiltonian import *

class testMMSTHamiltonian(object):
    def setup(self):
        # TODO: create a test for a 3-electronic by 2-nuclear dof system
        # for now, we only have thorough tests for 0 or 1 nuclear DOFs; the
        # monodromy matrices deserve better attention beyond that.
        from example_systems import tully
        self.tully = tully.potential
        self.tully_snap = tully.snapshots[0].copy()
        # [x0, x1, R] = [0.7, 0.6, 0.1]
        # [p0, p1, P] = [0.2, 0.1, 19.0]
        # tully_V11(tully_snap) = 0.1*tanh(1.6*0.1) = 0.0158648504297499
        # tully_V12(tully_snap) = 0.05*exp(-1.0*(0.1-0.0)^2) 
        #                       = 0.0495024916874584
        # tully_V22(tully_snap) = -0.0158648504297499

        four_state_matrix = dynq.NonadiabaticMatrix(
            [[0.00, 1.00, 0.75, 0.00],
             [1.00, 1.50, 0.00, 1.50],
             [0.75, 0.00, 0.50, 2.00],
             [0.00, 1.50, 2.00, -1.0]]
        )
        self.four_state = MMSTHamiltonian(four_state_matrix)
        four_state_topology = dynq.Topology(
            masses=[],
            potential=self.four_state
        )
        self.four_state_snap = dynq.MMSTSnapshot(
            coordinates=[],
            momenta=[],
            electronic_coordinates=np.array([0.5, 0.6, 0.7, 0.8]),
            electronic_momenta=np.array([0.1, 0.2, 0.3, 0.4]),
            topology=four_state_topology
        )

    def test_elect_cache(self):
        tully_elect = self.tully._elect_cache(self.tully_snap)
        # tully_elect[(0,0)] == 0.5*(0.7*0.7 + 0.2*0.2 - 1.0) = -0.235
        # tully_elect[(1,1)] == 0.5*(0.6*0.6 + 0.1*0.1 - 1.0) = -0.315
        # tully_elect[(0,1)] == 0.7*0.6 + 0.2*0.1 = 0.44
        assert_almost_equal(tully_elect[(0,0)], -0.235)
        assert_almost_equal(tully_elect[(1,1)], -0.315)
        assert_almost_equal(tully_elect[(0,1)], 0.44)
        if (1,0) in tully_elect.keys():
            raise AssertionError("Unexpected key in MMSTHamiltonian._elect")

        four_state_elect = self.four_state._elect_cache(self.four_state_snap)
        # four_state_elect[(0,0)] == 0.5*(0.5^2 + 0.1^2 - 1.0) = -0.37
        # four_state_elect[(0,1)] == 0.5*0.6 + 0.1*0.2 = 0.32
        # four_state_elect[(0,2)] == 0.5*0.7 + 0.1*0.3 = 0.38
        # four_state_elect[(0,3)] == 0.5*0.8 + 0.1*0.4 = 0.44
        # four_state_elect[(1,1)] == 0.5*(0.6^2 + 0.2^2 - 1.0) = -0.3
        # four_state_elect[(1,2)] == 0.6*0.7 + 0.2*0.3 = 0.48
        # four_state_elect[(1,3)] == 0.6*0.8 + 0.2*0.4 = 0.56
        # four_state_elect[(2,2)] == 0.5*(0.7^2 + 0.3^2 - 1.0) = -0.21
        # four_state_elect[(2,3)] == 0.7*0.8 + 0.3*0.4 = 0.68
        # four_state_elect[(3,3)] == 0.5*(0.8^2 + 0.4^2 - 1.0) = -0.1
        expected_elect = {(0,0) : -0.37, (0,1) : 0.32, (0,2) : 0.38, 
                          (0,3) : 0.44, (1,1) : -0.3, (1,2) : 0.48, 
                          (1,3) : 0.56, (2,2) : -0.21, (2,3) : 0.68, 
                          (3,3) : -0.1}
        assert_equal(set(four_state_elect.keys()), set(expected_elect.keys()))
        for k in four_state_elect.keys():
            assert_almost_equal(four_state_elect[k], expected_elect[k])


    def test_V(self):
        # V =  -0.235 * 0.0158648504297499 # V11
        #      -0.315 * -0.0158648504297499 # V22
        #      + 0.44 * 0.0495024916874584 # V12 
        #   = 0.0230502843768617
        assert_almost_equal(self.tully.V(self.tully_snap), 0.0230502843768617)
        
        # V =  -0.37*0.00 + 0.32*1.00 + 0.38*0.75 + 0.44*0.00
        #                 - 0.30*1.50 + 0.48*0.00 + 0.56*1.50
        #                             - 0.21*0.50 + 0.68*2.00
        #                                         - 0.10*-1.00
        #   = 2.35
        assert_almost_equal(self.four_state.V(self.four_state_snap), 2.35)
        

    def test_dHdq(self):
        # dV11dx = 0.155972904333467
        # dV22dx = -0.155972904333467
        # dV12dx = -0.00990049833749168
        # dHdq =  dV11dx*elect_11 + dV22dx*elect_22 + dV12dx*elect_12
        #      =   0.155972904333467 * -0.235
        #        -0.155972904333467 * -0.315
        #        -0.00990049833749168 * 0.44
        #      = 0.00812161307818102
        assert_array_almost_equal(self.tully.dHdq(self.tully_snap),
                                  0.00812161307818102)
        assert_array_almost_equal(self.four_state.dHdq(self.four_state_snap),
                                  np.array([]))

    def test_dHdp(self):
        assert_array_almost_equal(self.tully.dHdp(self.tully_snap), 
                                  np.array([19.0/1980.0]))
        assert_array_almost_equal(self.four_state.dHdp(self.four_state_snap),
                                  np.array([]))

    def test_electronic_dHdq(self):
        # dHdx1 = V11*x1 + V12*x2
        #       = 0.0158648504297499 * 0.7 + 0.0495024916874584 * 0.6
        #       = 0.0408068903133000
        # dHdx2 = V22*x2 + V12*x1
        #       = -0.0158648504297499 * 0.6 + 0.0495024916874584 * 0.7
        #       = 0.0251328339233709
        assert_array_almost_equal(self.tully.electronic_dHdq(self.tully_snap),
                                  np.array([0.0408068903133000,
                                            0.0251328339233709]))
        # dHdx0 = 0.00*0.5 + 1.00*0.6 + 0.75*0.7 + 0.00*0.8 = 1.125
        # dHdx1 = 1.00*0.5 + 1.50*0.6 + 0.00*0.7 + 1.50*0.8 = 2.60
        # dHdx2 = 0.75*0.5 + 0.00*0.6 + 0.50*0.7 + 2.00*0.8 = 2.325
        # dHdx3 = 0.00*0.5 + 1.50*0.6 + 2.00*0.7 + -1.0*0.8 = 1.5
        assert_array_almost_equal(
            self.four_state.electronic_dHdq(self.four_state_snap),
            np.array([1.125, 2.6, 2.325, 1.5])
        )

    def test_electronic_dHdp(self):
        # dHdp1 = V11*p1 + V12*p2
        #       = 0.0158648504297499 * 0.2 + 0.0495024916874584 * 0.1
        #       = 0.00812321925469582
        # dHdp2 = V22*p2 + V12*p1
        #       = -0.0158648504297499 * 0.1 + 0.0495024916874584 * 0.2
        #       = 0.00831401329451669
        assert_array_almost_equal(self.tully.electronic_dHdp(self.tully_snap),
                                  np.array([0.00812321925469582,
                                            0.00831401329451669]))
        # dHdp0 = 0.00*0.1 + 1.00*0.2 + 0.75*0.3 + 0.00*0.4 = 0.425
        # dHdp1 = 1.00*0.1 + 1.50*0.2 + 0.00*0.3 + 1.50*0.4 = 1.0
        # dHdp2 = 0.75*0.1 + 0.00*0.2 + 0.50*0.3 + 2.00*0.4 = 1.025
        # dHdp3 = 0.00*0.1 + 1.50*0.2 + 2.00*0.3 + -1.0*0.4 = 0.5
        assert_array_almost_equal(
            self.four_state.electronic_dHdp(self.four_state_snap),
            np.array([0.425, 1.0, 1.025, 0.5])
        )

    def test_T(self):
        # Definition of T is $L + V = p * dH/dp - H + V$; test this directly.
        # Use a lambda rather than nested def because nested def screws with
        # my in-editor test runner
        explicit_T = lambda pes, snap : (
            np.dot(snap.momenta, pes.dHdp(snap))
            + np.dot(snap.electronic_momenta, pes.electronic_dHdp(snap))
            - pes.H(snap) + pes.V(snap)
        )
        pes_T = self.tully.T(self.tully_snap)
        assert_almost_equal(pes_T, explicit_T(self.tully, self.tully_snap))
        assert_almost_equal(explicit_T(self.four_state, self.four_state_snap),
                            self.four_state.T(self.four_state_snap))

    def test_d2Hdq2(self):
        tully_d2Hdq2 = self.tully.d2Hdq2(self.tully_snap)
        V_ij = self.tully.H_matrix.numeric_matrix(self.tully_snap)
        assert_almost_equal(tully_d2Hdq2[0][0], V_ij[0][0])
        assert_almost_equal(tully_d2Hdq2[0][1], V_ij[0][1])
        assert_almost_equal(tully_d2Hdq2[1][0], V_ij[1][0])
        assert_almost_equal(tully_d2Hdq2[1][1], V_ij[1][1])
        
        runnables = self.tully.H_matrix.runnable_entries
        dVdx_ij = {}
        d2Vdx2_ij = {}
        for k in runnables.keys():
            dVdx_ij[k] = runnables[k].dHdq(self.tully_snap)
            d2Vdx2_ij[k] = runnables[k].d2Hdq2(self.tully_snap)

        x = self.tully_snap.electronic_coordinates
        p = self.tully_snap.electronic_momenta

        assert_almost_equal(tully_d2Hdq2[0][2],
                            x[0]*dVdx_ij[(0,0)] + x[1]*dVdx_ij[(0,1)])
        assert_almost_equal(tully_d2Hdq2[2][0],
                            x[0]*dVdx_ij[(0,0)] + x[1]*dVdx_ij[(1,0)])
        assert_almost_equal(tully_d2Hdq2[1][2],
                            x[0]*dVdx_ij[(1,0)] + x[1]*dVdx_ij[(1,1)])
        assert_almost_equal(tully_d2Hdq2[2][1],
                            x[0]*dVdx_ij[(0,1)] + x[1]*dVdx_ij[(1,1)])

        Hqq_22_00 = 0.5 * (x[0]*x[0] + p[0]*p[0] - 1.0) * d2Vdx2_ij[(0,0)]
        Hqq_22_11 = 0.5 * (x[1]*x[1] + p[1]*p[1] - 1.0) * d2Vdx2_ij[(1,1)]
        Hqq_22_01 = 0.5 * (x[0]*x[1] + p[0]*p[1]) * d2Vdx2_ij[(0,1)]
        Hqq_22_10 = 0.5 * (x[1]*x[0] + p[1]*p[0]) * d2Vdx2_ij[(1,0)]
        assert_almost_equal(
            tully_d2Hdq2[2][2], 
            (Hqq_22_00 + Hqq_22_01 + Hqq_22_10 + Hqq_22_11)[0][0]
        )

    def test_d2Hdp2(self):
        tully_d2Hdp2 = self.tully.d2Hdp2(self.tully_snap)
        V_ij = self.tully.H_matrix.numeric_matrix(self.tully_snap)
        # electronic-electronic
        assert_almost_equal(tully_d2Hdp2[0][0], V_ij[0][0])
        assert_almost_equal(tully_d2Hdp2[0][1], V_ij[0][1])
        assert_almost_equal(tully_d2Hdp2[1][0], V_ij[1][0])
        assert_almost_equal(tully_d2Hdp2[1][1], V_ij[1][1])

        # nuclear-nuclear
        assert_almost_equal(tully_d2Hdp2[2][2], 1.0/1980.0)

        # nuclear-electronic
        for i in [0, 1]:
            assert_equal(tully_d2Hdp2[i][2], 0.0)
            assert_equal(tully_d2Hdp2[2][i], 0.0)

    def test_d2Hdqdp_d2Hdpdq(self):
        tully_d2Hdqdp = self.tully.d2Hdqdp(self.tully_snap)
        tully_d2Hdpdq = self.tully.d2Hdpdq(self.tully_snap)

        # nuclear-electronic
        runnables = self.tully.H_matrix.runnable_entries
        dVdx_ij = {}
        for k in runnables.keys():
            dVdx_ij[k] = runnables[k].dHdq(self.tully_snap)
        p = self.tully_snap.electronic_momenta

        for i in range(3):
            for j in range(3):
                # assert symmetric; then only test Hqp
                assert_equal(tully_d2Hdqdp[(i,j)], tully_d2Hdpdq[(j,i)])
                if (i,j) == (2, 0):
                    assert_almost_equal(
                        tully_d2Hdqdp[(i,j)],
                        p[0]*dVdx_ij[(0,0)] + p[1]*dVdx_ij[(1,0)]
                    )
                elif (i,j) == (2, 1):
                    assert_almost_equal(
                        tully_d2Hdqdp[(i,j)],
                        p[0]*dVdx_ij[(0,1)] + p[1]*dVdx_ij[(1,1)]
                    )
                else:
                    assert_equal(tully_d2Hdqdp[(i,j)], 0.0)
