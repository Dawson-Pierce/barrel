"""Objects that emit physical phenomena detectable by BARREL sensors."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class DetectionEmission:
    source_id: object
    kind: str
    frame: str
    data: object
    metadata: dict = field(default_factory=dict)


class ObservedQuantityEmitter:
    """Expose an existing storm/latent observed quantity as optical emissions."""

    def __init__(self, quantity, rng=None, sample_dt_s=1.0, kind="optical_point"):
        self.quantity = quantity
        self.rng = rng or np.random.default_rng()
        self.sample_dt_s = float(sample_dt_s)
        self.kind = kind

    def emit(self, time_s):
        points = self.quantity.sample(float(time_s) / 60.0, self.rng, dt_s=self.sample_dt_s)
        return DetectionEmission(self.quantity.name, self.kind, "geodetic_lonlat", points)


class SatelliteSurfaceEmitter:
    """A spacecraft surface available to active ranging sensors."""

    def __init__(self, source_id, state, surface_points, metadata=None):
        self.source_id = source_id
        self.state = state
        self.surface_points = np.asarray(surface_points, dtype=float)
        self.metadata = dict(metadata or {})

    def emit(self, time_s):
        state = self.state(float(time_s)) if callable(self.state) else self.state
        return DetectionEmission(
            self.source_id,
            "reflective_surface",
            "eci",
            self.surface_points,
            {**self.metadata, "state": state},
        )

