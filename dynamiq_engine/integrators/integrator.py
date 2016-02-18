
# TODO: at some point we should change the integrators so they only alloc
# memory for every Nth time step, and also so they can alloc space for M
# snapshots in a seingle go (probably only once during initialization; then
# the results from those M snapshots can be reported (and copied, if needed)
# at once

class Integrator(object):
    monodromy = False
    def __init__(self, dt):
        self.dt = dt

    def step(self, snapshot):
        raise NotImplementedError("Abstract `Integrator` can't step.")

