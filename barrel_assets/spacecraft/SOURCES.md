# Spacecraft geometry and physical scale

The active five-template library contains Landsat 9, Hubble, Aqua, Chandra, and
Dawn. Each complete mesh has one scale in metres and one surface-area centroid.
Independent template and truth samples retain those units and that origin.
The origin is a geometric reference point, not a measured centre of mass.

| Spacecraft | Active STL | Scale reference | Exported body-coordinate spans in metres |
| --- | --- | --- | --- |
| Landsat 9 | `landsat9_assembled.stl` | 9.8 m solar-array length | 12.653, 4.090, 2.865 |
| Hubble | `hubble.stl` | 13.2 m axial length | 11.511, 13.200, 12.592 |
| Aqua | `aqua.stl` | 16.7 m deployed long span | 4.772, 8.831, 16.700 |
| Chandra | `chandra.stl` | Native metre-scale mesh, checked against 13.8 m by 19.5 m deployed size | 13.780, 5.138, 19.507 |
| Dawn | `dawn.stl` | Native metre-scale mesh, checked against 19.7 m deployed span | 19.698, 2.200, 2.215 |

These NASA visualization meshes approximate the spacecraft geometries. Uniform
scaling matches the stated reference dimension and preserves all mesh proportions.
Other dimensions are not independently fitted. In particular, Aqua's secondary
span is approximately 8.83 m in the mesh versus the published 8.0 m. These are
physically scaled geometric models, not engineering CAD or flight-vehicle models.

## Model sources

- [Assembled Landsat 9 model](https://science.nasa.gov/mission/landsat/outreach/landsat-9-3d-model/)
  supplies `landsat9_assembled.glb`. The array reference uses the combined extents
  of scene nodes `16`, `12`, `12_3`, and `11`, excluding the spacecraft bus.
- [Hubble Space Telescope A](https://github.com/nasa/NASA-3D-Resources/tree/master/3D%20Models/Hubble%20Space%20Telescope%20(A))
  supplies `hubble_a.glb`.
- [Aqua A](https://github.com/nasa/NASA-3D-Resources/tree/master/3D%20Models/Aqua%20(A))
  supplies `aqua.glb`.
- [Chandra X-ray Observatory](https://github.com/nasa/NASA-3D-Resources/tree/master/3D%20Models/Chandra%20X-ray%20Observatory)
  supplies `chandra.glb` and its existing complete-scene STL export.
- [Dawn](https://github.com/nasa/NASA-3D-Resources/tree/master/3D%20Models/Dawn)
  supplies `dawn.glb` and its existing complete-scene STL export.

## Dimensional references

- [NASA Landsat 9 fact sheet](https://www.nasa.gov/wp-content/uploads/2015/04/landsat_9_fast_facts.pdf)
- [NASA Hubble dimensions](https://science.nasa.gov/mission/hubble/overview/about-hubble/)
- [NASA Earth Science Reference Handbook, Aqua entry](https://eospso.gsfc.nasa.gov/sites/default/files/publications/2006ReferenceHandbook.pdf)
- [Chandra specifications](https://chandra.harvard.edu/about/specs.html)
- [NASA Dawn spacecraft specifications](https://science.nasa.gov/mission/dawn/technology/spacecraft/)

`physical_dimensions.json` preserves the original dimensional reference records.
`barrel_assets/nasa_sources.json` pins the five source URLs and SHA256 checksums;
the four GitHub URLs use a fixed NASA repository revision. No GLB/STL binaries
are stored in this repository. Run `python -m barrel_assets.nasa --download`
after installing `.[meshes]` to download only those sources and export complete
transformed scenes with `trimesh` and `DracoPy`. The generated output includes
conversion records for all five models. Source caches and generated meshes are
ignored by Git.

The companion Template_Matching_RFS application exposes the same converter via
`python -m tools.nasa_to_stl --download`, using its own ignored mesh directory.

The older `landsat9.stl` is a separated printing kit and `cubesat_mid.stl` is only
a printing section. They and `cygnss.stl` are not included or used by the current
simulations. Hubble and Aqua replace CYGNSS and CubeSat to retain
resolved returns at the existing range and angular resolution.
