# TODO: this will inherit from paths.DynamicsEngine when that gets moved to
# its own package -- maybe even before??
import openpathsampling as paths

from dynamiq_snapshot import Snapshot

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

