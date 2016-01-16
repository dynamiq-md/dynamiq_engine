# again, this should inherit from the paths.AbstractSnapshot
import openpathsampling as paths
import openpathsampling.features as features
import momentum_feature 

from openpathsampling.netcdfplus import StorableObject, lazy_loading_attributes

@lazy_loading_attributes('_reversed')
class Snapshot(paths.AbstractSnapshot):
    __features__ = [features.coordinates, momentum_feature]
    def __init__(self, coordinates=None, momenta=None, is_reversed=False,
                 topology=None, reversed_copy=None):
        """
        Creates a dynq.Snapshot
        """


    @property
    def coordinates(self):
        pass

    @property
    def velocities(self):
        pass

    @property
    def momenta(self):
        pass

    @property
    def n_atoms(self):
        pass

    xyz = coordinates

    def copy(self):
        this = Snapshot(
            coordinate=self.coordinates, 
            momenta=self.momenta, 
            is_reversed=self.is_reversed,
            topology=self.topology
        )

class Topology(paths.Topology):
    pass
