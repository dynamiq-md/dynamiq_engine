import dynamiq_engine as dynq
import numpy as np
from tools import *

from dynamiq_engine.integrators.candy_rozmus_4 import *
import dynamiq_engine.potentials as pes

class testCandyRozmus4(object):
    def setup(self):
        self.ho = dynq.potentials.interactions.HarmonicOscillatorInteraction(
            k=2.0, x0=1.0
        )
        self.potential = dynq.potentials.OneDimensionalInteractionModel(self.ho)
        self.topology = dynq.Topology(masses=np.array([0.5]), 
                                      potential=self.potential)
        self.integ = CandyRozmus4(0.01, self.potential)
        self.snap0 = dynq.Snapshot(
            coordinates=np.array([1.0]),
            momenta=np.array([0.0]),
            topology=self.topology
        )

    def exact_ho(self, initial_snap, time):
        m = initial_snap.topology.masses[0]
        omega = np.sqrt(self.ho.k / m)
        x0 = self.ho.x0
        p0 = initial_snap.momenta[0]
        q0 = initial_snap.coordinates[0]
        return exact_ho(time, omega, m, p0, q0, x0)

        #cos_wt = np.cos(omega*time)
        #sin_wt = np.sin(omega*time)
        #state_at_t = {
            #'q' : np.array([q0*cos_wt + p0/m/omega*sin_wt + x0]),
            #'p' : np.array([p0*cos_wt - q0*m*omega*sin_wt])
        #}
        #return state_at_t

    def test_cr4_step(self):
        new_snap = dynq.Snapshot(coordinates=np.array([0.0]),
                                 momenta=np.array([0.0]),
                                 topology=self.topology)
        # snap0 is at he minimum; it shouldn't move
        self.integ.step(self.potential, self.snap0, new_snap)
        assert_array_almost_equal(self.snap0.coordinates, new_snap.coordinates)
        assert_array_almost_equal(self.snap0.momenta, new_snap.momenta)
        for i in range(10):
            self.integ.step(self.potential, new_snap, new_snap)
        assert_array_almost_equal(self.snap0.coordinates, new_snap.coordinates)
        assert_array_almost_equal(self.snap0.momenta, new_snap.momenta)

        snap1 = dynq.Snapshot(coordinates=np.array([0.0]),
                              momenta=np.array([0.0]),
                              topology=self.topology)
        self.integ.step(self.potential, snap1, new_snap)
        exact_0x01 = self.exact_ho(snap1, 0.01)
        assert_array_almost_equal(new_snap.coordinates, exact_0x01['q'])
        assert_array_almost_equal(new_snap.momenta, exact_0x01['p'])

        for i in range(9):
            self.integ.step(self.potential, new_snap, new_snap)
        exact_0x10 = self.exact_ho(snap1, 0.10)
        assert_array_almost_equal(new_snap.coordinates, exact_0x10['q'])
        assert_array_almost_equal(new_snap.momenta, exact_0x10['p'])

class testCandyRozmus4MMST(object):
    def setup(self):
        tully_V11 = pes.OneDimensionalInteractionModel(
            pes.interactions.TanhInteraction(a=1.6, V0=0.1)
        )
        tully_V22 = pes.OneDimensionalInteractionModel(
            pes.interactions.TanhInteraction(a=1.6, V0=-0.1)
        )
        tully_V12 = pes.OneDimensionalInteractionModel(
            pes.interactions.GaussianInteraction(A=0.05, alpha=1.0)
        )
        tully_matrix = dynq.NonadiabaticMatrix([[tully_V11, tully_V12],
                                                [tully_V12, tully_V22]])
        self.tully = dynq.potentials.MMSTHamiltonian(tully_matrix)
        tully_topology = dynq.Topology(
            masses=np.array([1980.0]),
            potential=self.tully
        )

        self.tully_integ = CandyRozmus4MMST(1.0, self.tully)

        uncoupled_matrix = dynq.NonadiabaticMatrix([[2.0, 0.0], [0.0, 3.0]])
        self.uncoupled = dynq.potentials.MMSTHamiltonian(uncoupled_matrix)

        rabi_matrix = dynq.NonadiabaticMatrix([[2.0, 1.0], [1.0, 3.0]])
        self.rabi = dynq.potentials.MMSTHamiltonian(rabi_matrix)

        pass

    def test_cr4_step(self):
        from math import sqrt
        # test uncoupled
        uncoupled_topology = dynq.Topology(masses=[], potential=self.uncoupled)
        uncoupled_snap = dynq.MMSTSnapshot(
            coordinates=np.array([]), momenta=np.array([]),
            electronic_coordinates=np.array([1.0, 1.0]),
            electronic_momenta=np.array([1.0, 1.0]),
            topology=uncoupled_topology
        )
        uncoupled_integ = CandyRozmus4MMST(0.01, self.uncoupled)
        for i in range(10):
            uncoupled_integ.step(self.uncoupled, uncoupled_snap,
                                 uncoupled_snap)

        exact_1 = exact_ho(time=0.1, omega=2.0, m=1.0/2.0, q0=1.0, p0=1.0)
        exact_2 = exact_ho(time=0.1, omega=3.0, m=1.0/3.0, q0=1.0, p0=1.0)
        predicted_coordinates = [exact_1['q'][0], exact_2['q'][0]]
        predicted_momenta = [exact_1['p'][0], exact_2['p'][0]]
        assert_array_almost_equal(uncoupled_snap.electronic_coordinates,
                                  np.array(predicted_coordinates))
        assert_array_almost_equal(uncoupled_snap.electronic_momenta,
                                  np.array(predicted_momenta))

        # test Rabi
        #rabi_topology = dynq.Topology(masses=[], potential=self.rabi)
        #rabi_snapshot = dynq.MMSTSnapshot(
            #coordinates=np.array([]), momenta=np.array([]),
            #electronic_coordinates=np.array([1.0, 0.0]),
            #electronic_momenta=np.array([0.0, 1.0]),
            #topology=rabi_topology
        #)
        #rabi_integ = CandyRozmus4MMST(0.01, self.rabi)
        #for i in range(10):
            #rabi_integ.step(self.rabi, rabi_snapshot, rabi_snapshot)

        # TODO: test results


        # test Tully
        raise SkipTest
