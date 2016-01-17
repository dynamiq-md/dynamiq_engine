# again, this should inherit from the paths.AbstractSnapshot
import openpathsampling as paths
import openpathsampling.features as features
import momentum_feature 

from openpathsampling.netcdfplus import StorableObject, lazy_loading_attributes

class DynamiqEngine(paths.DynamicsEngine):
    default_options = {
        'integ' : None,
        'n_frames_max' : None,
        'nsteps_per_frame' : 1
    }
    def __init__(self, potential, integrator, template):
        self.potential = potential
        self.integrator = integrator

    @property
    def current_snapshot(self):
        pass

    @current_snapshot.setter
    def current_snapshot(self, snap):
        pass

    def generate_next_frame(self):
        self.integrator.step(self, self.nsteps_per_frame)
        return self.current_snapshot

@lazy_loading_attributes('_reversed')
class Snapshot(paths.AbstractSnapshot):
    __features__ = [features.coordinates, momentum_feature]
    def __init__(self, coordinates=None, momenta=None, is_reversed=False,
                 topology=None, reversed_copy=None):
        """
        Creates a dynq.Snapshot
        """
        self.coordinates = coordinates
        self.momenta = momenta


    @property
    def velocities(self):
        return self.momenta * self.topology.inverse_masses

    xyz = coordinates

    def copy(self):
        this = Snapshot(
            coordinate=self.coordinates, 
            momenta=self.momenta, 
            is_reversed=self.is_reversed,
            topology=self.topology
        )
        return this


class Topology(paths.Topology):
    def __init__(self, masses, potential):
        n_atoms = potential.n_atoms
        n_spatial = potential.n_spatial
        super(Topology, self).__init__(n_atoms, n_spatial)
        self.masses = masses
        self._inverse_masses = np.array([1.0/m for m in masses])
        self.potential = potential

    @property
    def inverse_masses(self):
        return self._inverse_masses

    def subset(self, list_of_atoms):
        return self
