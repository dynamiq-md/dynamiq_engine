import dynamiq_engine as dynq
import dynamiq_engine.potentials as pes
import numpy as np
from tools import *

from dynamiq_engine.potentials.mmst_hamiltonian import *

class testMMSTHamiltonian(object):
    def setup(self):
        tully_V11 = pes.OneDimensionalInteractionModel(
            pes.interactions.TanhInteraction(a=1.6, V0=0.1)
        )
        tully_V22 = pes.OneDimensionalInteractionModel(
            pes.interactions.TanhInteraction(a=1.6, V0=-0.1)
        )
        tully_V12 = pes.OneDimensionalInteractionModel(
            pes.interactions.GaussianInteraction(A=0.05, alpha=1.0)
        )
        tully_matrix = dynq.NonadiabaticMatrix([[tully_V11, tully_V12],
                                                [tully_V12, tully_V22]])
        self.tully = MMSTHamiltonian(tully_matrix)
        tully_topology = dynq.Topology(
            masses=np.array([1.0]),
            potential=tully_V11 #TODO: don't pick an arbitrary one...
        )

        self.tully_snap = dynq.MMSTSnapshot(
            coordinates=np.array([0.1]),
            momenta=np.array([19.0]),
            electronic_coordinates=np.array([0.7, 0.6]),
            electronic_momenta=np.array([0.2, 0.1]),
            topology=tully_topology
        )
        # tully_V11(tully_snap) = 0.1*tanh(1.6*0.1) = 0.0158648504297499
        # tully_V12(tully_snap) = 0.05*exp(-1.0*(0.1-0.0)^2) 
        #                       = 0.0495024916874584
        # tully_V22(tully_snap) = -0.0158648504297499

        #four_state_matrix = dynq.NonadiabaticMatrix()
        #four_state = MMSTHamiltonian(four_state_matrix)
        #four_state_snap = 
        pass

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

    def test_V(self):
        # V =  -0.235 * 0.0158648504297499 # V11
        #      -0.315 * -0.0158648504297499 # V22
        #      + 0.44 * 0.0495024916874584 # V12 
        #   = 0.0230502843768617
        assert_almost_equal(self.tully.V(self.tully_snap), 0.0230502843768617)

    def test_set_dHdq(self):
        pass

    def test_set_dHdp(self):
        pass

    def test_set_electonic_dHdq(self):
        pass

    def test_set_electonic_dHdp(self):
        pass

