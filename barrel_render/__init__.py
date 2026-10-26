"""barrel_render — Earth-centered Blender rendering for BARREL simulations.

Importable in two ways:
  * plain Python (no Blender): ``from barrel_render import data`` for the
    bpy-free CSV/astronomy helpers (unit-testable).
  * inside Blender: ``from barrel_render import EarthScene`` for the scene API.
"""

from . import data  # always available (no bpy dependency)

try:
    import bpy  # noqa: F401
    from . import materials  # noqa: F401
    from .scene import EarthScene  # noqa: F401
except ImportError:
    EarthScene = None  # running outside Blender; only `data` is available

__all__ = ["data", "materials", "EarthScene"]
