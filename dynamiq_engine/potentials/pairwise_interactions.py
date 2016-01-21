
class PairwiseInteraction(object):
    def f(self, x):
        raise NotImplementedError()

    def dfdx(self, x):
        raise NotImplementedError()

    def d2fdx2(self, x):
        raise NotImplementedError()

    def __call__(self, x):
        return self.f(x)

    # these are only relevant if you have atom-atom distances to calculate
    def pdot_part(self, snapshot, i, j, pdot):
        pass #pragma: no cover

    def V_part(self, snapshot, i, j):
        pass #pragma: no cover

class ConstantInteraction(PairwiseInteraction):
    def __init__(self, value):
        self.value = value

    def f(self, x):
        return self.value

    def dfdx(self, x):
        return 0.0

    def d2fdx2(self, x):
        return 0.0

class HarmonicOscillatorInteraction(PairwiseInteraction):
    def __init__(self, k, x0):
        self.k = k
        self.x0 = x0

    def f(self, x):
        return 0.5*self.k*(x-self.x0)*(x-self.x0)

    def dfdx(self, x):
        return self.k*(x-self.x0)

    def d2fdx2(self, x):
        return self.k



