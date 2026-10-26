"""Resolve generated spacecraft meshes or user-supplied mesh paths."""

from __future__ import annotations

from pathlib import Path
import os

ASSET_DIR = Path(os.environ.get("BARREL_ASSET_DIR", "output/meshes")).resolve()

_ALIASES = {
    "cubesat": "cubesat_mid.stl",
    "cubesat_mid": "cubesat_mid.stl",
    "cygnss": "cygnss.stl",
    "landsat": "landsat9_assembled.stl",
    "landsat9": "landsat9_assembled.stl",
    "hubble": "hubble.stl",
    "aqua": "aqua.stl",
    "chandra": "chandra.stl",
    "dawn": "dawn.stl",
}


def available_assets():
    """Return the names that can be passed to ``EarthScene.add_satellites(asset=...)``."""
    return sorted(_ALIASES)


def asset_path(name):
    """Resolve a named spacecraft asset or direct filesystem path."""
    if name is None:
        return None
    p = Path(str(name))
    if p.exists():
        return str(p)
    key = str(name).lower().strip()
    if key not in _ALIASES:
        raise KeyError(f"unknown spacecraft asset {name!r}; available: {', '.join(available_assets())}")
    resolved = ASSET_DIR / _ALIASES[key]
    if not resolved.exists():
        raise FileNotFoundError(f"{resolved}: prepare NASA meshes or set BARREL_ASSET_DIR to your mesh directory")
    return str(resolved)
