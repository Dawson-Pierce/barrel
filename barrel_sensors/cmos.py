"""Nadir CMOS space sensor: a satellite observes ground points (e.g. lightning flashes)
through a pinhole CMOS array. Each visible point is projected to the image plane, ROUNDED
to its CMOS pixel (the dominant measurement error), then the pixel centre is remapped back
through a ray -> WGS-84 ellipsoid -> ECEF -> geodetic (lon, lat) for a GLOBAL tracking
frame. Misses (1 - Pd) and uniform pixel clutter are injected.

Everything is in ECEF: the satellite ECI position is rotated to ECEF (GMST) and a nadir
camera frame (x=east, y=north, z=zenith; boresight = -z) is built there. Pure NumPy."""

import numpy as np

from barrel_simulation import coordinates as bdata
from .base import MeasurementBatch, Sensor

_A = 6378137.0
_F = 1.0 / 298.257223563
_B = _A * (1.0 - _F)
_E2 = _F * (2.0 - _F)
_EP2 = _E2 / (1.0 - _E2)


def ecef_to_geodetic(p):
    """ECEF metres -> (lon_deg, lat_deg)."""
    x, y, z = p
    lon = np.arctan2(y, x)
    pr = np.hypot(x, y)
    th = np.arctan2(z * _A, pr * _B)
    lat = np.arctan2(z + _EP2 * _B * np.sin(th) ** 3,
                     pr - _E2 * _A * np.cos(th) ** 3)
    return np.degrees(lon), np.degrees(lat)


def _ray_ellipsoid(o, d):
    """Nearest intersection t of ray o + t d with the WGS-84 ellipsoid, or None."""
    s = np.array([1.0 / _A, 1.0 / _A, 1.0 / _B])
    op, dp = o * s, d * s
    A, Bc, C = dp @ dp, op @ dp, op @ op - 1.0
    disc = Bc * Bc - A * C
    if disc < 0.0 or A == 0.0:
        return None
    t = (-Bc - np.sqrt(disc)) / A
    return t if t > 0.0 else None


class CmosSensor(Sensor):
    """Nadir CMOS imager: pinhole projection + pixel quantization + ground remap."""

    def __init__(self, pixels=512, fov_deg=18.0, prob_detection=0.9, clutter_rate=1.0,
                 max_incidence_deg=70.0, name="cmos", cadence_s=0.0):
        super().__init__(name=name, cadence_s=cadence_s)
        self.W = int(pixels)
        self.fx = (self.W / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
        self.cx = self.W / 2.0
        self.pd = prob_detection
        self.clutter_rate = clutter_rate
        # Observe a ground point only where the satellite is within this angle of the
        # local zenith -- grazing limb views (near the horizon) geolocate poorly and the
        # far side is occluded. Bounds each observer to a near-nadir cap of the globe.
        self.cos_inc_min = np.cos(np.radians(max_incidence_deg))

    def acquire(self, scene, host, time_s, rng):
        """Acquire every optical-point emission currently present in ``scene``."""
        emissions = scene.emissions("optical_point")
        points = [np.asarray(item.data, dtype=float) for item in emissions if np.asarray(item.data).size]
        truth = np.hstack(points) if points else np.zeros((2, 0))
        measured = self.observe(host.position_eci, host.epoch_jd, truth, rng)
        return self.record(MeasurementBatch(
            self.name, time_s, measured, "geodetic_lonlat",
            tuple(item.source_id for item in emissions), {"truth_count": int(truth.shape[1])},
        ))

    @staticmethod
    def _camera_frame(sat_ecef):
        zen = sat_ecef / np.linalg.norm(sat_ecef)
        east = np.array([-zen[1], zen[0], 0.0])
        n = np.linalg.norm(east)
        east = east / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
        north = np.array([zen[1] * east[2] - zen[2] * east[1],
                          zen[2] * east[0] - zen[0] * east[2],
                          zen[0] * east[1] - zen[1] * east[0]])
        return np.column_stack([east, north, zen])  # cam x, y, z(zenith); boresight = -z

    def _project(self, R, sat_ecef, pt_ecef):
        pc = R.T @ (pt_ecef - sat_ecef)
        depth = -pc[2]
        if depth <= 0.0:
            return None
        u = self.fx * pc[0] / depth + self.cx
        v = self.cx - self.fx * pc[1] / depth
        return (u, v) if (0.0 <= u < self.W and 0.0 <= v < self.W) else None

    def _remap_pixel(self, R, sat_ecef, u, v):
        d_cam = np.array([(u - self.cx) / self.fx, -(v - self.cx) / self.fx, -1.0])
        d_cam /= np.linalg.norm(d_cam)
        d = R @ d_cam
        t = _ray_ellipsoid(sat_ecef, d)
        return None if t is None else ecef_to_geodetic(sat_ecef + t * d)

    def camera_frame_ecef(self, sat_pos_eci, jd):
        """(sat_ecef, R) for the nadir camera at a logged ECI position + epoch."""
        sat_ecef = np.asarray(bdata.eci_to_ecef(tuple(sat_pos_eci), jd), dtype=float)
        return sat_ecef, self._camera_frame(sat_ecef)

    @staticmethod
    def ground_points_ecef(points_lonlat):
        return np.asarray([
            bdata.geodetic_to_ecef(np.radians(lat), np.radians(lon), 0.0)
            for lon, lat in points_lonlat.T], dtype=float).reshape(-1, 3)

    def observe(self, sat_pos_eci, jd, points_lonlat, rng, *, camera_frame=None,
                points_ecef=None):
        """Return (2, M) lon/lat 'space-observed' detections for one satellite at one
        frame. points_lonlat is the (2, N) truth ground points (lon, lat)."""
        sat_ecef, R = (self.camera_frame_ecef(sat_pos_eci, jd)
                       if camera_frame is None else camera_frame)
        points = (self.ground_points_ecef(points_lonlat)
                  if points_ecef is None else points_ecef)
        los_all = sat_ecef - points
        radial = points / np.linalg.norm(points, axis=1, keepdims=True)
        # Cull clearly invisible points in a batch. Borderline candidates still
        # pass through the original scalar gate and projection below.
        candidates = np.flatnonzero(
            np.sum(los_all * radial, axis=1)
            >= self.cos_inc_min * np.linalg.norm(los_all, axis=1) - 1e-6)
        out = []
        for j in candidates:
            pt = points[j]
            los = sat_ecef - pt
            # Near-side + min-elevation gate (excludes the occluded far side and grazing
            # limb views that geolocate poorly).
            if float(los @ (pt / np.linalg.norm(pt))) < self.cos_inc_min * np.linalg.norm(los):
                continue
            uv = self._project(R, sat_ecef, pt)
            if uv is None or rng.random() > self.pd:
                continue
            meas = self._remap_pixel(R, sat_ecef, np.floor(uv[0]) + 0.5, np.floor(uv[1]) + 0.5)
            if meas is not None:
                out.append(meas)
        for _ in range(int(rng.poisson(self.clutter_rate))):
            meas = self._remap_pixel(R, sat_ecef, rng.uniform(0, self.W), rng.uniform(0, self.W))
            if meas is not None:
                out.append(meas)
        return np.array(out).T if out else np.zeros((2, 0))
