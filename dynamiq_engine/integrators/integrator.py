
class Integrator(object):
    monodromy = False
    def __init__(self, dt):
        self.dt = dt

    def step(self, snapshot):
        raise NotImplementedError("Abstract `Integrator` can't step.")
