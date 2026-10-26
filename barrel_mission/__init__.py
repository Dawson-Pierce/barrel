"""barrel_mission -- a single Python class that merges the BARREL dynamics and renderer:
own a list of satellites, command them (pointing, maneuvers, camera tasking), propagate the
dynamics (Basilisk astrodynamics: spherical-harmonic gravity, drag, Sun/Moon third body, and
reaction-wheel attitude control), log CSVs, and render into a run-scoped output folder.
"""

from .mission import Mission
from .constellation import WalkerDeltaConstellation

__all__ = ["Mission", "WalkerDeltaConstellation"]
