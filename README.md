# BARREL

BARREL runs spacecraft simulations: dynamics, sampled trajectories, physical
sources, sensors, estimator chains, and scheduled communications. Applications
supply experiment settings, estimator adapters, and result processing. BREW
remains the numerical estimation library, accessed through brewpy.

## One simulation owner

```python
import numpy as np
from barrel_simulation import Simulation, WalkerDeltaConstellation

shell = WalkerDeltaConstellation(n_planes=2, sats_per_plane=3)
with Simulation(name="example", output_root="output", log_dt=1) as simulation:
    for elements in shell.elements():
        simulation.add_satellite(elements=elements)
    simulation.run(np.arange(61), record=lambda sim: print(sim.state(0)["position"]))
```

`Simulation` owns the clock, spacecraft, sources, payloads, and event schedule.
`run(times, record=...)` executes absolute sample times in seconds. The optional
record callback receives the simulation after sensing and scheduled actions.
`step_to(time)` advances one sample. `command(satellite, ...)` changes guidance or
applies an impulsive delta-v to a dynamically propagated spacecraft. Satellites
can be identified by their returned object, name, or list index.

Basilisk implementation details live in `_dynamics.py`. There is no separate
public Mission, Environment, or Scene. `Satellite` is the one spacecraft object
for both live propagation and stored ephemerides. Sample histories and current
state belong to that same object.

## Precomputed constellations

```python
shell = WalkerDeltaConstellation(n_planes=8, sats_per_plane=12)
ephemeris = shell.propagate(minutes=720, dt_s=1, dynamics_dt_s=10,
                            workers=12, cache_root="cache/sim")
simulation = Simulation(seed=3)
for index in range(shell.satellite_count):
    simulation.add_ephemeris(ephemeris["t_s"], ephemeris["jd"],
                             ephemeris["pos"][:, index],
                             plane=index // shell.sats_per_plane,
                             slot=index % shell.sats_per_plane)
```

Walker propagation retains the ten-second orbital integration, cubic Hermite
interpolation, and separate one-second reaction-wheel/control pass. Worker count
does not change cache identity. `dynamics_dt_s=1` selects coupled propagation.
The separate passes assume constant drag area and balanced wheels.

The constellation result currently exposes positions rather than controlled
attitudes. `add_ephemeris` uses ideal nadir attitude unless quaternions are supplied.
Sampled histories interpolate position linearly and use the nearest recorded
attitude; commands require live Basilisk spacecraft. `circular()` remains an
inexpensive two-body alternative. `ring_geometry()` evaluates adjacent links.

## Sources, sensors, and estimator chains

Sources implement `emit(time_s)` and return `DetectionEmission` objects.
`ObservedQuantityEmitter` adapts synthetic optical point processes;
`SatelliteSurfaceEmitter` supplies spacecraft surfaces. Register either with
`simulation.add_source(source)`. A source is evaluated once per acquisition epoch
so all sensors share the same physical realization.

Attach a `SensorChannel` to the spacecraft returned by `add_satellite` or
`add_ephemeris`. A channel contains one sensor and an ordered set of estimators:

```python
from barrel_sensors import SensorChannel, EstimatorNode

# lidar, pose_estimator, and intent_estimator are configured by the application.
channel = SensorChannel(lidar, [
    EstimatorNode("pose", pose_estimator),
    EstimatorNode("intent", intent_estimator, source="pose"),
])
spacecraft.add_payload(channel)
```

An estimator implements `predict(time_s, dt_s, host=None)`,
`correct(value, time_s, host=None)`, and `estimate()`. Root nodes receive a
`MeasurementBatch`; downstream nodes receive the named upstream estimate.
Multiple nodes may consume the same upstream result. `channel.outputs` exposes
the latest estimates. Prediction intervals follow the sensor's acquisition
cadence, including when other simulation events occur between acquisitions.

Use `CmosSensor`, `LidarSensor`, or `Sensor(sample=callback, frame="...")` to
connect a study-specific acquisition model. The sampling callback receives
`(simulation, spacecraft, time_s, rng)`. Template Matching uses this interface
for BARREL mesh ray tracing and its BREW pose estimator.

## Scheduling and communications

`simulation.schedule(name, times, action)` runs `action(simulation)` at the
specified absolute times. At a shared time, sensors run first, actions run in
registration order, and the record callback runs last. Events between sample
times advance dynamics without acquiring sensors. `simulation.events` contains
the action return values produced during the current step.

This supports commands, accumulated onboard updates, fusion, and downlinks.
`SatelliteFusionRing` routes estimates using an injected numerical fusion
function. `GroundStation` receives ring products and updates an attached estimator.
The application specifies schedules and fusion policy; BARREL executes them.

## Layout

- `barrel_simulation/simulation.py`: Simulation and Satellite.
- `barrel_simulation/_dynamics.py`: internal Basilisk force/control implementation.
- `barrel_simulation/constellation.py`: Walker shells, propagation, and caches.
- `barrel_simulation/sources.py`: emissions and synthetic source models.
- `barrel_simulation/network.py`: ring routing and ground estimation.
- `barrel_simulation/coordinates.py`: numerical coordinate/astronomy utilities.
- `barrel_sensors/`: sensors, measurement batches, estimator nodes, mesh lidar.
- `barrel_assets/`: NASA downloads, STL conversion, and surface sampling.
- `tests/`: shared framework regression tests.

Blender rendering, stereo reconstruction, duplicate environment/runtime wrappers,
legacy guidance wrappers, and rendering examples were removed. Numerical coordinate
helpers and all sensor geometry used by the journals remain. Old source was backed
up outside the dissertation workspace before removal.

## Install and Docker

```sh
python -m pip install -e '.[meshes,lidar]'
python -m unittest discover -s tests -p 'test_*.py'
docker build -t barrel .
docker run --rm barrel
```

Python 3.10+ is required. Basilisk is pinned to 2.9.1. The two application images
build their own brewpy/BREW extension and include latexmk for their journals.

## NASA geometry

```sh
python -m barrel_assets.nasa --download --assets cache/nasa --output output/meshes
```

Only five selected, checksum-pinned NASA models are downloaded and converted.
`--models landsat9 hubble` selects a subset. Source transforms and physical scale
are preserved. Set `BARREL_ASSET_DIR` to resolve generated mesh aliases elsewhere.
No meshes, textures, simulation results, or compiled journals belong in Git.
