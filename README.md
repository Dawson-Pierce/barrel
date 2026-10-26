# BARREL

BARREL is the shared spacecraft simulation framework. It owns environments and
scene timing, satellites and guidance, sampled geometry, sensors and estimator
interfaces, communication/fusion rings, ground stations, and Basilisk propagation
and logging. Estimation applications attach their own BREW trackers to it.

The small optional Blender rendering API and numerical astronomy helpers remain.
Large texture collections and spacecraft binaries are not stored in this repo.
The default Docker image does not install Blender or download spacecraft models.

## Install

Use Python 3.10 or newer:

```sh
python -m pip install -e .
```

This installs NumPy, Basilisk (`bsk==2.9.1`), and pytest, which Basilisk imports
through its runtime support code. For NASA geometry preparation also install:

```sh
python -m pip install -e '.[meshes]'
```

## Simulation

```python
from barrel_mission import Mission
from barrel_environment import Scene
from barrel_sensors import CmosSensor, LidarSensor

with Mission(name="example", output_root="output") as mission:
    mission.add_satellite(elements=(6928137.0, 0.001, 0.9, 0.0, 0.0, 0.0))
    mission.run(duration=60.0, dt=1.0)
    state = mission.state(0)
```

The small gravity coefficient file is packaged with BARREL. Numerical propagation,
sensing, scene timing, guidance, communications, and fusion implementations are
unchanged by the asset cleanup.

## NASA geometry

```sh
python -m barrel_assets.nasa --download
```

This downloads only the five selected sources to `cache/nasa/` and writes scaled
STLs plus conversion metadata to `output/meshes/`. Both directories are ignored.
`--models landsat9 hubble` selects a subset; `--assets` and `--output` change the
cache and output locations. Omit `--download` to convert an existing verified cache.

Four models come from a fixed revision of NASA-3D-Resources; the assembled
Landsat 9 source comes from NASA's mission site. Source SHA256 checksums are
recorded in `barrel_assets/nasa_sources.json` and verified before conversion.
The conversion preserves the original scene transforms and physical scaling.
See `barrel_assets/spacecraft/SOURCES.md` for model and dimensional references.
No entire NASA repository is cloned.

`asset_path()` resolves generated models under `output/meshes/`, or the directory
specified by `BARREL_ASSET_DIR`. Explicit mesh paths are also accepted. Legacy
CubeSat/CYGNSS aliases require user-supplied files. Textures and Blender can be
provided separately when optional rendering is wanted; sensor simulation does
not require them.

## Docker

The default image includes the numerical framework and optional mesh-conversion
libraries, with no bundled meshes or textures:

```sh
docker build -t barrel .
docker run --rm barrel
docker run --rm -v "${PWD}/output:/work/output" barrel python -m barrel_assets.nasa --download
```

The two application Dockerfiles supply their own pinned scientific packages.
Template_Matching_RFS prepares its required meshes during its image build;
storm_global_estimator does not download geometry or external weather data.

## Layout

- `barrel_environment/`: environments, satellites, event-driven scenes, emitters,
  guidance interfaces, communication rings, and ground stations.
- `barrel_mission/`: Basilisk orbital/attitude propagation, controls, and CSV logs.
- `barrel_sensors/`: sensor/estimator interfaces, CMOS, lidar, and stereo geometry.
- `barrel_assets/`: asset lookup, verified NASA downloads, STL conversion, metadata.
- `barrel_render/`: shared numerical helpers and optional Blender scene code.
- `tests/`: runtime and NASA source-integrity tests.

```sh
python -m unittest discover -s tests -p 'test_*.py'
```
