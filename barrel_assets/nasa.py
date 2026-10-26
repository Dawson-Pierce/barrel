"""Fetch verified NASA GLB sources and convert complete spacecraft meshes to metres."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

SOURCES = json.loads(Path(__file__).with_name("nasa_sources.json").read_text())


@dataclass(frozen=True)
class MeshSpec:
    source: str
    output: str
    reference: str
    dimension: float | None = None
    nodes: tuple[str, ...] | None = None
    axis: int | None = None


MODELS = {
    "landsat9": MeshSpec("landsat9_assembled.glb", "landsat9_assembled.stl", "solar-array length",
                        9.8, ("16", "12", "12_3", "11"), 0),
    "hubble": MeshSpec("hubble_a.glb", "hubble.stl", "axial length", 13.2, None, 1),
    "aqua": MeshSpec("aqua.glb", "aqua.stl", "deployed long span", 16.7, None, 2),
    "chandra": MeshSpec("chandra.glb", "chandra.stl", "native metre-scale complete scene"),
    "dawn": MeshSpec("dawn.glb", "dawn.stl", "native metre-scale complete scene"),
}


class NasaMeshConverter:
    """Preserve scene transforms, then apply one documented physical scale."""

    def __init__(self, assets="cache/nasa", output="output/meshes"):
        self.assets, self.output = Path(assets), Path(output)

    def download(self, models=tuple(MODELS)):
        """Download only selected models; accept cached files only at the recorded hash."""
        self.assets.mkdir(parents=True, exist_ok=True)
        for name in models:
            spec = SOURCES[name]
            target = self.assets / spec["source"]
            if target.exists():
                self._verified_source(name)
                continue
            request = Request(spec["url"], headers={"User-Agent": "BARREL-NASA-meshes"})
            with urlopen(request, timeout=120) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != spec["sha256"]:
                raise ValueError(f"NASA source checksum mismatch: {name}; downloaded data was not saved")
            target.write_bytes(data)
            print(f"Downloaded {name}")

    def _verified_source(self, name):
        spec = SOURCES[name]
        source = self.assets / spec["source"]
        if not source.is_file():
            raise FileNotFoundError(f"Missing {source}; rerun the mesh preparation command with --download")
        if hashlib.sha256(source.read_bytes()).hexdigest() != spec["sha256"]:
            raise ValueError(f"NASA source checksum mismatch: {source}")
        return source

    def convert(self, models=tuple(MODELS)):
        import DracoPy  # Required before trimesh reads Draco-compressed GLB scenes.
        import numpy as np
        import trimesh
        self.output.mkdir(parents=True, exist_ok=True)
        records = {}
        for name in models:
            spec = MODELS[name]
            source = self._verified_source(name)
            scene = trimesh.load(source)
            mesh = scene.to_mesh()
            raw_span, scale = None, 1.0
            if spec.dimension is not None:
                if spec.nodes is None:
                    reference_vertices = mesh.vertices
                else:
                    reference_vertices = np.vstack([
                        trimesh.transform_points(scene.geometry[scene.graph[node][1]].vertices,
                                                 scene.graph[node][0]) for node in spec.nodes])
                raw_span = float(np.ptp(reference_vertices[:, spec.axis]))
                scale = spec.dimension / raw_span
                mesh.apply_scale(scale)
            target = self.output / spec.output
            mesh.export(target)
            records[spec.output] = dict(
                source=spec.source, reference=spec.reference,
                reference_dimension_m=spec.dimension, reference_axis=spec.axis,
                reference_nodes=spec.nodes, raw_reference_span=raw_span,
                metres_per_source_unit=scale, exported_span_m=mesh.extents.tolist(),
                triangles=len(mesh.faces),
                source_url=SOURCES[name]["url"],
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                stl_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
            print(spec.output, "span [m]", mesh.extents.round(4), "scale", scale)
        (self.output / "physical_dimensions.json").write_text(json.dumps(records, indent=2) + "\n")
        return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=Path("cache/nasa"))
    parser.add_argument("--output", type=Path, default=Path("output/meshes"))
    parser.add_argument("--download", action="store_true", help="Download verified NASA sources into the local cache")
    parser.add_argument("--models", nargs="+", choices=tuple(MODELS), default=tuple(MODELS))
    args = parser.parse_args()
    converter = NasaMeshConverter(args.assets, args.output)
    if args.download:
        converter.download(args.models)
    converter.convert(args.models)


if __name__ == "__main__":
    main()
