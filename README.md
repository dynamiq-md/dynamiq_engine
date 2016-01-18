[![Build Status](https://travis-ci.org/dwhswenson/dynamiq_engine.svg?branch=master)](https://travis-ci.org/dwhswenson/dynamiq_engine)
[![Coverage
Status](https://coveralls.io/repos/dwhswenson/dynamiq_engine/badge.svg?branch=master&service=github)](https://coveralls.io/github/dwhswenson/dynamiq_engine?branch=master)

# DynamiqEngine

MD Engine for `dynamiq` (engine as a separate package for reuse elsewhere)

The overall `dynamiq` package involves two parts. The first part is
`dynamiq` itself, which is (will be) a Python package to do various versions
of the semiclassical initial value representation (e.g., HK-IVR, FB-IVR)
that require propagation of the monodromy matrix.

The second part is this package, `dynamiq_engine`, which provides the
molecular dynamics engine for doing such propagations. This engine is
compatible with OpenPathSampling engines, making it reusable for many other
projects. The engine uses (will use) Cython to optimize speed.

The OPS-compatibility code is mostly in `dynamiq_engine_core.py`, and
includes an `DynamicsEngine` subclass, a `Snapshot`, and a `Topology`. There
are also several new snapshot features in the `snapshot_features` directory.

The actual engine code is in the `integrators` and `potentials` directories. 
