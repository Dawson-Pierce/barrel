"""Active lidar model producing partial point clouds from spacecraft surfaces."""

from __future__ import annotations

import numpy as np

from .base import MeasurementBatch, Sensor


def _rotation(state):
    if isinstance(state, dict):
        return np.asarray(state.get("rotation", np.eye(3)), dtype=float)
    return np.asarray(getattr(state, "rotation", np.eye(3)), dtype=float)


def _position(state):
    if isinstance(state, dict):
        return np.asarray(state.get("position", state.get("pos")), dtype=float)
    return np.asarray(getattr(state, "position", getattr(state, "pos", None)), dtype=float)


class LidarSensor(Sensor):
    """Beam-cell lidar with front-surface occlusion, range noise, and missed returns."""

    def __init__(self, name="lidar", cadence_s=0.0, pixels=52, max_range_m=5000.0,
                 fov_deg=20.0, range_sigma_m=0.02, cross_range_sigma_m=0.01,
                 prob_detection=0.98, mount_dcm=None):
        super().__init__(name=name, cadence_s=cadence_s)
        self.pixels = int(pixels)
        self.max_range_m = float(max_range_m)
        self.tan_half_fov = np.tan(np.radians(float(fov_deg)) / 2.0)
        self.range_sigma_m = float(range_sigma_m)
        self.cross_range_sigma_m = float(cross_range_sigma_m)
        self.prob_detection = float(prob_detection)
        self.mount_dcm = np.eye(3) if mount_dcm is None else np.asarray(mount_dcm, dtype=float)

    def acquire(self, scene, host, time_s, rng):
        origin = np.asarray(host.position_eci, dtype=float)
        R_si = np.asarray(host.sensor_rotation_eci(self), dtype=float)  # sensor axes -> ECI
        clouds, sources = [], []
        for emission in scene.emissions("reflective_surface"):
            if emission.source_id == host.sat_id:
                continue
            state = emission.metadata["state"]
            center = _position(state)
            if center is None or np.linalg.norm(center - origin) > self.max_range_m:
                continue
            surface = np.asarray(emission.data, dtype=float)
            world = (_rotation(state) @ surface.T).T + center
            rel = (world - origin) @ R_si
            cloud = self._scan(rel, rng)
            if len(cloud):
                clouds.append(cloud @ R_si.T + origin)
                sources.append(emission.source_id)
        data = np.vstack(clouds).T if clouds else np.zeros((3, 0))
        return self.record(MeasurementBatch(self.name, time_s, data, "eci", tuple(sources)))

    def _scan(self, rel, rng):
        # Sensor +Z is the lidar boresight. Retain the nearest return in each angular cell.
        z = rel[:, 2]
        valid = (z > 0.0) & (z <= self.max_range_m)
        xang = np.divide(rel[:, 0], z, out=np.zeros_like(z), where=z != 0.0)
        yang = np.divide(rel[:, 1], z, out=np.zeros_like(z), where=z != 0.0)
        valid &= (np.abs(xang) <= self.tan_half_fov) & (np.abs(yang) <= self.tan_half_fov)
        idx = np.flatnonzero(valid)
        if not len(idx):
            return np.zeros((0, 3))
        u = np.clip(((xang[idx] / self.tan_half_fov + 1.0) * 0.5 * self.pixels).astype(int), 0, self.pixels - 1)
        v = np.clip(((yang[idx] / self.tan_half_fov + 1.0) * 0.5 * self.pixels).astype(int), 0, self.pixels - 1)
        order = idx[np.argsort(z[idx])]
        keys = u[np.argsort(z[idx])] * self.pixels + v[np.argsort(z[idx])]
        _, first = np.unique(keys, return_index=True)
        hit = rel[order[first]].copy()
        keep = rng.random(len(hit)) <= self.prob_detection
        hit = hit[keep]
        if len(hit):
            hit[:, 2] += rng.normal(0.0, self.range_sigma_m, len(hit))
            hit[:, :2] += rng.normal(0.0, self.cross_range_sigma_m, (len(hit), 2))
        return hit
