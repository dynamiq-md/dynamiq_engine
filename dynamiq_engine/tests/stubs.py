import dynamiq_engine as dynq

class PotentialStub(dynq.potentials.PotentialEnergySurface):
    def H(self, snapshot):
        return 0.0

    def dHdq(self, snapshot):
        return 0.0

    def dHdp(self, snapshot):
        return 0.0
