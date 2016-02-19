import dynamiq_engine as dynq
import numpy as np

class ExampleSystem(object):
    def __init__(self, potential, integrator, masses):
        self.potential = potential
        self.integrator = integrator
        self.topology = dynq.Topology(masses=masses,
                                      potential=self.potential)
        self.snapshots = []

    def add_example_snapshots(self, snapshots):
        self.snapshots.extend(snapshots)
        pass

def make_tully():
    tully_V11 = dynq.potentials.OneDimensionalInteractionModel(
        dynq.potentials.interactions.TanhInteraction(a=1.6, V0=0.1)
    )
    tully_V22 = dynq.potentials.OneDimensionalInteractionModel(
        dynq.potentials.interactions.TanhInteraction(a=1.6, V0=-0.1)
    )
    tully_V12 = dynq.potentials.OneDimensionalInteractionModel(
        dynq.potentials.interactions.GaussianInteraction(A=0.05, alpha=1.0)
    )
    tully_matrix = dynq.NonadiabaticMatrix([[tully_V11, tully_V12],
                                            [tully_V12, tully_V22]])
    tully = dynq.potentials.MMSTHamiltonian(tully_matrix)
    masses=np.array([1980.0]),
    integrator = dynq.integrators.CandyRozmus4(dt=0.01, potential=tully)
    sys = ExampleSystem(potential=tully, integrator=integrator, masses=masses)
    sys.add_example_snapshots([
        dynq.MMSTSnapshot(
            coordinates=np.array([0.1]),
            momenta=np.array([19.0]),
            electronic_coordinates=np.array([0.7, 0.6]),
            electronic_momenta=np.array([0.2, 0.1]),
            topology=sys.topology
        )
        # [x0, x1, R] = [0.7, 0.6, 0.1]
        # [p0, p1, P] = [0.2, 0.1, 19.0]
    ])
    return sys

tully = make_tully()

def make_anharmonic_morse():
    potential = dynq.potentials.OneDimensionalInteractionModel(
        dynq.potentials.interactions.MorseInteraction(D=30.0, beta=0.08, x0=0.5)
    )
    topology = dynq.Topology(masses=np.array([1.0]),
                             potential=potential)
    integrator = dynq.integrators.CandyRozmus4(dt=0.1, potential=potential)
    anharmonic_morse = ExampleSystem(
        potential=potential,
        topology=topology,
        integrator=integrator
    )
    return anharmonic_morse

anharmonic_morse = make_anharmonic_morse()
