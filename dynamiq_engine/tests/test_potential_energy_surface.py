import dynamiq_engine as dynq
import numpy as np
from tools import *
import dynamiq_engine.potentials as pes

from dynamiq_engine.potentials.potential_energy_surface import *

class testOneDimensionalInteractionModel(object):
    def setup(self):
        from example_systems import ho_2_1
        self.pot = ho_2_1.potential

# TODO: we keep this around because I think we intend to merge some other
# stuff into it
