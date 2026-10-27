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



@dataclass
class GeodeticEllipseSet:
    """Moving geodetic ellipse point process.

    Storm dictionaries use longitude and latitude in degrees, optional motion
    in degrees per minute, extent standard deviations in degrees, and ``gamma``
    as the Poisson mean per ``reference_dt_s`` seconds.
    """

    name: str
    storms: list[dict]
    reference_dt_s: float = 30.0
    lon_start: float | None = None
    lon_rate: float = 0.0
    lat_clip: tuple[float, float] = (-89.0, 89.0)
    _last_samples: np.ndarray = field(default_factory=lambda: np.zeros((2, 0)), init=False)

    @staticmethod
    def _wrap_lon(lon):
        return ((float(lon) + 180.0) % 360.0) - 180.0

    @staticmethod
    def _spherical_state(storm, epoch_min):
        lon = np.radians(float(storm["lon"]))
        lat = np.radians(float(storm["lat"]))
        heading = np.radians(float(storm["heading_deg"]))
        angle = float(storm["angular_speed_rad_min"]) * float(epoch_min)

        radial = np.array([
            np.cos(lat) * np.cos(lon),
            np.cos(lat) * np.sin(lon),
            np.sin(lat),
        ])
        east = np.array([-np.sin(lon), np.cos(lon), 0.0])
        north = np.array([
            -np.sin(lat) * np.cos(lon),
            -np.sin(lat) * np.sin(lon),
            np.cos(lat),
        ])
        tangent = np.sin(heading) * east + np.cos(heading) * north
        propagated_radial = np.cos(angle) * radial + np.sin(angle) * tangent
        propagated_tangent = -np.sin(angle) * radial + np.cos(angle) * tangent
        return propagated_radial, propagated_tangent

    def centroids(self, epoch_min):
        centers = np.zeros((2, len(self.storms)), dtype=float)
        for index, storm in enumerate(self.storms):
            if "angular_speed_rad_min" in storm and "heading_deg" in storm:
                radial, _ = self._spherical_state(storm, epoch_min)
                lon = np.degrees(np.arctan2(radial[1], radial[0]))
                lat = np.degrees(np.arcsin(np.clip(radial[2], -1.0, 1.0)))
            elif "lon" in storm:
                lon = storm["lon"] + storm.get("dlon", 0.0) * float(epoch_min)
                lat = storm["lat"] + storm.get("dlat", 0.0) * float(epoch_min)
            elif self.lon_start is not None:
                lon = self.lon_start + self.lon_rate * float(epoch_min)
                lat = storm["lat"] + storm.get("dlat", 0.0) * float(epoch_min)
            else:
                raise ValueError("storm needs 'lon' or the ellipse set needs lon_start")
            centers[0, index] = self._wrap_lon(lon)
            centers[1, index] = np.clip(lat, *self.lat_clip)
        return centers

    def tangent_velocities(self, epoch_min):
        """Return east/north angular velocity in radians per minute."""
        centers = self.centroids(epoch_min)
        velocities = np.zeros((len(self.storms), 2), dtype=float)
        for index, storm in enumerate(self.storms):
            if "angular_speed_rad_min" in storm and "heading_deg" in storm:
                radial, tangent = self._spherical_state(storm, epoch_min)
                lon = np.arctan2(radial[1], radial[0])
                lat = np.arcsin(np.clip(radial[2], -1.0, 1.0))
                east = np.array([-np.sin(lon), np.cos(lon), 0.0])
                north = np.array([
                    -np.sin(lat) * np.cos(lon),
                    -np.sin(lat) * np.sin(lon),
                    np.cos(lat),
                ])
                angular_speed = float(storm["angular_speed_rad_min"])
                velocities[index] = angular_speed * np.array([
                    tangent @ east,
                    tangent @ north,
                ])
            else:
                lat = np.radians(centers[1, index])
                velocities[index] = [
                    np.cos(lat) * np.radians(
                        float(storm.get("dlon", self.lon_rate))),
                    np.radians(float(storm.get("dlat", 0.0))),
                ]
        return velocities

    def truth_frame(self, epoch_min):
        centers = self.centroids(epoch_min)
        return [
            (centers[:, index].copy(),
             np.diag(np.square(np.asarray(storm["ext"], dtype=float))),
             float(storm["gamma"]))
            for index, storm in enumerate(self.storms)
        ]

    def sample(self, epoch_min, rng, dt_s=1.0):
        centers = self.centroids(epoch_min)
        points = []
        for index, storm in enumerate(self.storms):
            count = rng.poisson(
                float(storm["gamma"]) * float(dt_s) / float(self.reference_dt_s))
            if count:
                extent_lon, extent_lat = storm["ext"]
                longitude = centers[0, index] + rng.normal(0.0, float(extent_lon), count)
                latitude = np.clip(
                    centers[1, index] + rng.normal(0.0, float(extent_lat), count),
                    *self.lat_clip)
                points.append(np.column_stack((longitude, latitude)))
        self._last_samples = np.vstack(points).T if points else np.zeros((2, 0))
        return self._last_samples

