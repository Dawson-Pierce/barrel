"""Physical quantities observed by BARREL sensors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


class ObservedQuantity:
    """Base interface for quantities that live in a BARREL environment."""

    name: str

    def truth_frame(self, epoch_min):
        raise NotImplementedError

    def sample(self, epoch_min, rng, dt_s=1.0):
        raise NotImplementedError

    def render(self, scene, epoch_min=0.0):
        return []


@dataclass
class CallableObservedQuantity(ObservedQuantity):
    """Adapt existing truth and sampling functions to an observed quantity."""

    name: str
    truth_fn: Callable
    sample_fn: Callable
    render_fn: Callable | None = None

    def truth_frame(self, epoch_min):
        return self.truth_fn(epoch_min)

    def sample(self, epoch_min, rng, dt_s=1.0):
        return self.sample_fn(epoch_min, rng, dt_s)

    def render(self, scene, epoch_min=0.0):
        return [] if self.render_fn is None else self.render_fn(scene, epoch_min)


@dataclass
class GeodeticEllipseSet(ObservedQuantity):
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
    render_color: tuple[float, float, float] = (1.0, 0.9, 0.25)
    render_width: float = 0.008
    render_sigma: float = 2.0
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

    def ellipse_lonlat(self, mean, extent, n=96):
        values, vectors = np.linalg.eigh(np.asarray(extent, dtype=float))
        theta = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=True)
        axes = self.render_sigma * np.sqrt(np.maximum(values, 1e-9))
        local = vectors @ np.vstack((axes[0] * np.cos(theta), axes[1] * np.sin(theta)))
        lonlat = np.asarray(mean, dtype=float)[:, None] + local
        lonlat[0] = (lonlat[0] + 180.0) % 360.0 - 180.0
        lonlat[1] = np.clip(lonlat[1], *self.lat_clip)
        return lonlat.T

    @staticmethod
    def _as_scene_geodetic(lonlat):
        return [(float(row[1]), float(row[0]), 0.0) for row in np.asarray(lonlat)]

    def render(self, scene, epoch_min=0.0):
        objects = [
            scene.add_line(
                self._as_scene_geodetic(self.ellipse_lonlat(mean, extent)),
                frame="geodetic", color=self.render_color,
                width=self.render_width, strength=3.5)
            for mean, extent, _ in self.truth_frame(epoch_min)
        ]
        if self._last_samples.shape[1]:
            objects.append(scene.add_lightning(
                self._as_scene_geodetic(self._last_samples.T),
                frame="geodetic", size=0.025))
        return objects
