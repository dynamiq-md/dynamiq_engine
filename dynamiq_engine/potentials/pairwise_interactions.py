
class PairwiseInteraction(object):
    def f(self, x):
        raise NotImplementedError()

    def dfdx(self, x):
        raise NotImplementedError()

    def d2fdx2(self, x):
        raise NotImplementedError()

    def pdot_part(self, snapshot, i, j, pdot):
        pass

    def V_part(self, snapshot, i, j):
        pass

class HarmonicOscillatorInteraction(PairwiseInteraction):
    def __init__(self, k, x0):
        self.k = k
        self.x0 = x0

    def f(self, x):
        return 0.5*k*(x-x0)*(x-x0)

    def dfdx(self, x):
        return k*(x-x0)

    def d2fdx2(self, x):
        return k



