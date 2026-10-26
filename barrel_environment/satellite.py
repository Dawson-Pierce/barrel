"""Runtime satellite, payload, and guidance/control objects."""

from __future__ import annotations

import numpy as np


class GuidanceControlSystem:
    def __init__(self, mode="nadir"):
        self.mode = mode
        self.target_latlon = None
        self.target_eci = None

    def command(self, scene, satellite, mode=None, target_latlon=None, target_eci=None, delta_v_eci=None):
        if mode is not None:
            self.mode = mode
        if target_latlon is not None:
            self.mode, self.target_latlon = "target", target_latlon
        if target_eci is not None:
            self.mode, self.target_eci = "target", target_eci
        if scene.dynamics is not None and hasattr(scene.dynamics, "command"):
            scene.dynamics.command(satellite.dynamics_id, pointing=self.mode,
                                   target_latlon=target_latlon, target_eci=target_eci,
                                   delta_v_eci=delta_v_eci)


class Satellite:
    """Live spacecraft state with attached GNC and sensor/estimator channels."""

    def __init__(self, sat_id, dynamics_id=None, guidance_control=None):
        self.sat_id = sat_id
        self.dynamics_id = sat_id if dynamics_id is None else dynamics_id
        self.guidance_control = guidance_control or GuidanceControlSystem()
        self.payloads = []
        self.position_eci = np.zeros(3)
        self.velocity_eci = np.zeros(3)
        self.rotation_eci = np.eye(3)
        self.jd = 0.0

    def add_payload(self, channel):
        self.payloads.append(channel)
        return channel

    def sync_truth(self, scene):
        if scene.dynamics is None or not hasattr(scene.dynamics, "state"):
            return
        state = scene.dynamics.state(self.dynamics_id)
        self.position_eci = np.asarray(state["position"], dtype=float)
        self.velocity_eci = np.asarray(state.get("velocity", np.zeros(3)), dtype=float)
        self.rotation_eci = np.asarray(state.get("rotation", np.eye(3)), dtype=float)
        self.jd = float(state.get("jd", self.jd))

    def sensor_rotation_eci(self, sensor):
        return self.rotation_eci @ np.asarray(getattr(sensor, "mount_dcm", np.eye(3)), dtype=float)

    def acquire(self, scene, time_s, dt_s, rng):
        return [channel.acquire_and_update(scene, self, time_s, dt_s, rng)
                for channel in self.payloads if channel.sensor.is_due(time_s)]
