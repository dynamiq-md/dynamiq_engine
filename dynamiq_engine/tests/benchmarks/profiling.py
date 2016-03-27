import dynamiq_engine as dynq
import openpathsampling.engines as peng
import numpy as np

def run_engine(engine, snapshot, ntraj=1, nsteps=1000):
    for t in range(ntraj):
        engine.start()
        length_criteria = lambda x, foo : len(x) <= nsteps
        trajectory = engine.generate(snapshot, running=[length_criteria])

def morse_setup():
    # Morse test system: a test of 1D performance
    potential = dynq.potentials.interactions.MorseInteraction(D=30.0,
                                                              beta=0.08,
                                                              x0=0.0)
    integrator = dynq.integrators.CandyRozmus4(dt=0.1, potential=potential)
    masses = [1.0]
    topology = dynq.Topology(masses=[1.0], potential=potential)
    snapshot = dynq.Snapshot(coordinates=np.array([0.0]),
                             momenta=np.array([0.5]),
                             topology=topology)
    engine = dynq.DynamiqEngine(potential, integrator, template=snapshot)
    return engine, snapshot

def morse_run(ntrajs, nsteps):
    engine, snapshot = morse_setup()
    run_engine(engine, snapshot, ntrajs, nsteps)

cprofile_tests = {
    'morse_run(ntrajs=1, nsteps=1000)': "morse_1_1000.pstats",
    'morse_run(ntrajs=100, nsteps=10)': "morse_100_10.pstats"
}

if __name__ == "__main__":
    import cProfile
    for k in cprofile_tests:
        cProfile.run(k, cprofile_tests[k])
