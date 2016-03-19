import openpathsampling.engines.features as paths_f
import dynamiq_engine.features as dynq_f
import dynamiq_engine as dynq
import numpy as np
# NOTE: there are two ways to calculate the monodromy matrix: either you
# calculate it in some form of M = M + dM/dt*t, or you calculate it directly
# (by finite difference or by M(t0, t2) = M(t0,t1) * M(t1,t2)). 
# Actually, this should be 3 ways, with different `monodromy_type`s in the
# specific `MonodromyHelper`s:
#   1. `dt`
#   2. `fd`
#   3. `local` (as with G&L's unitary propagation)
# All of these also create `monodromy`, but the existence of each lets the
# integrator know which way to do things. Use supported features on each to
# ensure that we get the right thing.
class MonodromyHelper(object):
    monodromy_type = None
    def prepare(self, integrator):
        self.n_dim = integrator.potential.n_dofs 
        pass

    def reset(self, initial_snapshot):
        # we use Mqq as a proxy for the whole monodromy matrix: Mqq is only
        # not None if all of them are not
        if initial_snapshot.Mqq is not None:
            initial_snapshot.Mqq.fill(0.0)
            initial_snapshot.Mpp.fill(0.0)
            for i in range(self.n_dim):
                initial_snapshot.Mqq[(i,i)] = 1.0
                initial_snapshot.Mpp[(i,i)] = 1.0
            initial_snapshot.Mqp.fill(0.0)
            initial_snapshot.Mpq.fill(0.0)
        else:
            initial_snapshot.Mqq = np.identity(self.n_dim)
            initial_snapshot.Mpp = np.identity(self.n_dim)
            initial_snapshot.Mqp = np.zeros((self.n_dim, self.n_dim))
            initial_snapshot.Mpq = np.zeros((self.n_dim, self.n_dim))

class StandardMonodromy(MonodromyHelper):
    monodromy_type = "dt"

    def __init__(self, second_derivatives=None):
        self.second_derivatives = second_derivatives
        # second derivatives allows us to override, for example, the
        # implementation of the Hessian
        self.cross_terms = True # default always correct, even if rare


    def prepare(self, integrator):
        super(StandardMonodromy, self).prepare(integrator)
        if self.second_derivatives is None:
            self.second_derivatives = integrator.potential

        n_dim = self.n_dim
        self._local_dMqq_dt = np.zeros((n_dim, n_dim))
        self._local_dMqp_dt = np.zeros((n_dim, n_dim))
        self._local_dMpq_dt = np.zeros((n_dim, n_dim))
        self._local_dMpp_dt = np.zeros((n_dim, n_dim))

        # TODO: these second derivatives will be cached
        self._local_Hqq = np.zeros((n_dim, n_dim))
        self._local_Hpp = np.zeros((n_dim, n_dim))
        self._local_Hqp = None
        self._local_Hpq = None
        if self.second_derivatives.cross_terms:
            self._tmp = np.zeros((n_dim, n_dim))
            self._local_Hqp = np.zeros((n_dim, n_dim))
            self._local_Hpq = np.zeros((n_dim, n_dim))


    def dMqq_dt(self, potential, snapshot):
        """dMqq_dt = Hpp*Mpq + Hpq*Mqq"""
        self.second_derivatives.set_d2Hdp2(self._local_Hpp, snapshot)
        np.dot(self._local_Hpp, snapshot.Mpq, out=self._local_dMqq_dt)

        if self.second_derivatives.cross_terms:
            self.second_derivatives.set_d2Hdpdq(self._local_Hpq, snapshot)
            np.dot(self._local_Hpq, snapshot.Mqq, out=self._tmp)
            np.add(self._local_dMqq_dt, self._tmp, out=self._local_dMqq_dt)

        return self._local_dMqq_dt


    def dMqp_dt(self, potential, snapshot):
        """dMqp_dt = Hpq*Mqp + Hpp*Mpp"""
        self.second_derivatives.set_d2Hdp2(self._local_Hpp, snapshot)
        np.dot(self._local_Hpp, snapshot.Mpp, out=self._local_dMqp_dt)

        if self.second_derivatives.cross_terms:
            self.second_derivatives.set_d2Hdpdq(self._local_Hpq, snapshot)
            np.dot(self._local_Hpq, snapshot.Mqp, out=self._tmp)
            np.add(self._local_dMqp_dt, self._tmp, out=self._local_dMqp_dt)

        return self._local_dMqp_dt


    def dMpq_dt(self, potential, snapshot):
        """dMpq_dt = -Hqq*Mqq - Hqp*Mpq"""
        self.second_derivatives.set_d2Hdq2(self._local_Hqq, snapshot)
        np.dot(-self._local_Hqq, snapshot.Mqq, out=self._local_dMpq_dt)

        if self.second_derivatives.cross_terms:
            self.second_derivatives.set_d2Hdqdp(self._local_Hqp, snapshot)
            np.dot(self._local_Hqp, snapshot.Mpq, out=self._tmp)
            np.add(self._local_dMpq_dt, self._tmp, out=self._local_dMpq_dt)

        return self._local_dMpq_dt

    def dMpp_dt(self, potential, snapshot):
        """dMpp_dt = -Hqq*Mqp - Hqp*Mpp"""
        self.second_derivatives.set_d2Hdq2(self._local_Hqq, snapshot)
        np.dot(-self._local_Hqq, snapshot.Mqp, out=self._local_dMpp_dt)

        if self.second_derivatives.cross_terms:
            self.second_derivatives.set_d2Hdqdp(self._local_Hqp, snapshot)
            np.dot(self._local_Hqp, snapshot.Mpp, out=self._tmp)
            np.add(self._local_dMpp_dt, self._tmp, self._local_dMpp_dt)

        return self._local_dMpp_dt


class FiniteDifferenceMonodromy(MonodromyHelper):
    monodromy_type = "fd"
    def __init__(self, deltas=None):
        pass

    def reset(self, initial_snapshot):
        pass

    def update_monodromy(self, snapshot):
        pass


class GarashchukLightMonodromy(FiniteDifferenceMonodromy):
    monodromy_type = "local"
    def __init__(self, deltas=None):
        pass

    def reset(self, initial_snapshot):
        pass

    def update_monodromy(self, snapshot):
        pass
