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
        pass

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

