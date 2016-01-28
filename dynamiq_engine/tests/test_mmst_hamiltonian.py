import dynamiq_engine as dynq
import dynamiq_engine.potentials as pes
import numpy as np
from tools import *

from dynamiq_engine.potentials.mmst_hamiltonian import *

class testMMSTHamiltonian(object):
    def setup(self):
        tully_V11 = pes.interactions.TanhInteraction(a=1.6, V0=0.1)
        tully_V22 = pes.interactions.TanhInteraction(a=1.6, V0=-0.1)
        tully_V12 = pes.interactions.GaussianInteraction(A=0.05, alpha=1.0)
        tully_matrix = dynq.NonadiabaticMatrix([[tully_V11, tully_V12],
                                                [tully_V12, tully_V22]])
        self.tully = MMSTHamiltonian(tully_matrix)
        self.tully_snap = dynq.MMSTSnapshot(
            coordinates=np.array([0.1]),
            momenta=np.array([19.0]),
            electronic_coordinates=np.array([0.7, 0.6]),
            electronic_momenta=np.array([0.2, 0.1])
            # can we get away with no topology?
        )
        # tully_V11(tully_snap) = 
        # tully_V12(tully_snap) = 
        # tully_V22(tully_snap) = 

        #four_state_matrix = dynq.NonadiabaticMatrix()
        #four_state = MMSTHamiltonian(four_state_matrix)
        #four_state_snap = 
        pass

    def test_elect_cache(self):
        tully_elect = self.tully._elect_cache(self.tully_snap)
        assert_almost_equal(tully_elect[(0,0)], 0.5*(0.7*0.7 + 0.2*0.2 - 1.0))
        assert_almost_equal(tully_elect[(1,1)], 0.5*(0.6*0.6 + 0.1*0.1 - 1.0))
        assert_almost_equal(tully_elect[(0,1)], 0.7*0.6 + 0.2*0.1)
        if (1,0) in tully_elect.keys():
            raise AssertionError("Unexpected key in MMSTHamiltonian._elect")

    def test_H(self):
        pass

    def test_V(self):
        pass

    def test_set_dHdq(self):
        pass

    def test_set_dHdp(self):
        pass

    def test_set_electonic_dHdq(self):
        pass

    def test_set_electonic_dHdp(self):
        pass

