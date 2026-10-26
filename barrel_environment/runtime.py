"""Event-driven BARREL scene runtime."""

from __future__ import annotations

import numpy as np


def _quat_xyzw_to_dcm(q):
    x, y, z, w = np.asarray(q, dtype=float)
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


class EphemerisDynamics:
    """Event-driven adapter for an Environment's existing sampled ephemerides."""

    def __init__(self, environment):
        self.environment = environment
        self.time_s = 0.0

    def step(self, dt_s):
        self.time_s += float(dt_s)

    def state(self, sat):
        s = self.environment.satellites[int(sat)]
        k = int(np.argmin(np.abs(s.t - self.time_s)))
        if len(s.t) > 1:
            j = min(k + 1, len(s.t) - 1) if s.t[k] <= self.time_s else max(k - 1, 0)
            if j != k:
                alpha = np.clip((self.time_s - s.t[k]) / (s.t[j] - s.t[k]), 0.0, 1.0)
                pos = (1.0 - alpha) * s.pos[k] + alpha * s.pos[j]
            else:
                pos = s.pos[k]
            velocity = np.gradient(s.pos, s.t, axis=0, edge_order=1)[k]
        else:
            pos, velocity = s.pos[k], np.zeros(3)
        return {"position": pos, "velocity": velocity,
                "rotation": _quat_xyzw_to_dcm(s.quat[k]), "jd": float(s.jd[k]),
                "time_s": self.time_s}


class Scene:
    """Physical world advanced to discrete acquisition, control, or render events."""

    def __init__(self, dynamics=None, environment=None, seed=0):
        self.dynamics = dynamics
        self.environment = environment
        self.time_s = 0.0
        self.satellites = []
        self.fusion_rings = []
        self.ground_stations = []
        self.detectables = []
        self.rng = np.random.default_rng(seed)
        self._emissions = None

    @classmethod
    def from_environment(cls, environment, seed=0):
        return cls(EphemerisDynamics(environment), environment, seed=seed)

    def add_satellite(self, satellite):
        self.satellites.append(satellite)
        return satellite

    def add_detectable(self, obj):
        self.detectables.append(obj)
        return obj

    def add_fusion_ring(self, ring):
        self.fusion_rings.append(ring)
        return ring

    def add_ground_station(self, station):
        self.ground_stations.append(station)
        return station

    def route_ring(self, ring, start_node, ground_station, time_s=None):
        """Execute one Pierced-scheduled ring handoff and deliver it to ground."""
        if ring not in self.fusion_rings:
            raise ValueError("fusion ring is not registered with this scene")
        if ground_station not in self.ground_stations:
            raise ValueError("ground station is not registered with this scene")
        product = ring.fuse(start_node, self.time_s if time_s is None else time_s)
        ground_station.receive(product)
        return product

    def emissions(self, kind=None):
        if self._emissions is None:
            self._emissions = []
            for obj in self.detectables:
                emitted = obj.emit(self.time_s)
                if emitted is None:
                    continue
                items = emitted if isinstance(emitted, (list, tuple)) else [emitted]
                self._emissions.extend(items)
        return [x for x in self._emissions if kind is None or x.kind == kind]

    def step_to(self, time_s, acquire=True):
        target = float(time_s)
        if target < self.time_s:
            raise ValueError("scene time cannot move backward")
        dt_s = target - self.time_s
        if self.dynamics is not None and dt_s > 0.0:
            self.dynamics.step(dt_s)
        self.time_s = target
        self._emissions = None  # one physical realization shared by every sensor at this epoch
        for sat in self.satellites:
            sat.sync_truth(self)
        if acquire:
            for sat in self.satellites:
                sat.acquire(self, self.time_s, dt_s, self.rng)
        return self

    def render(self, **kwargs):
        if self.dynamics is not None and hasattr(self.dynamics, "render"):
            return self.dynamics.render(**kwargs)
        if self.environment is not None:
            return self.environment.render_still(frame=-1, **kwargs)
        raise RuntimeError("scene has no renderer")
