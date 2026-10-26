"""BARREL environment: physical satellites, propagation sources, assets, and scene state.

This layer coordinates BARREL pieces without importing dissertation ``sim`` or ``brewpy``
code. It is meant to own spacecraft geometry/orbit/render state; higher-level simulations
can attach estimators or verification logic outside BARREL.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from barrel_assets import asset_path
from barrel_render import data as bdata

MU_EARTH = 3.986004418e14
R_EARTH = 6378137.0
JD0 = 2461043.5


@dataclass
class EnvironmentSatellite:
    """Physical spacecraft state used by BARREL rendering and sensor geometry."""

    sat_id: int
    plane: int
    slot: int
    t: np.ndarray
    jd: np.ndarray
    pos: np.ndarray
    quat: np.ndarray
    name: str = ""
    asset: str | None = None

    def trajectory_record(self):
        return {
            "t": self.t,
            "jd": self.jd,
            "pos": [tuple(row) for row in self.pos],
            "quat": [tuple(row) for row in self.quat],
        }


def orbit_period_min(alt_km=550.0):
    a = R_EARTH + float(alt_km) * 1e3
    return 2.0 * math.pi * math.sqrt(a**3 / MU_EARTH) / 60.0


def walker_angles(plane, slot, n_planes, sats_per_plane, phasing=1):
    total = int(n_planes) * int(sats_per_plane)
    raan = 2.0 * math.pi * int(plane) / float(n_planes)
    nu0 = 2.0 * math.pi * int(slot) / float(sats_per_plane)
    nu0 += 2.0 * math.pi * float(phasing) * int(plane) / float(total)
    return raan, nu0 % (2.0 * math.pi)


def circular_orbit_history(
    plane,
    slot,
    n_planes,
    sats_per_plane,
    minutes,
    dt_s,
    alt_km=550.0,
    inc_deg=53.0,
    phasing=1,
):
    """Circular two-body Walker satellite history in ECI metres."""
    a = R_EARTH + float(alt_km) * 1e3
    inc = math.radians(float(inc_deg))
    raan, nu0 = walker_angles(plane, slot, n_planes, sats_per_plane, phasing)
    n = math.sqrt(MU_EARTH / a**3)
    times = np.arange(0.0, float(minutes) * 60.0 + 0.5 * float(dt_s), float(dt_s))
    jd = JD0 + times / 86400.0

    ci, si = math.cos(inc), math.sin(inc)
    cr, sr = math.cos(raan), math.sin(raan)
    u = nu0 + n * times
    cu = np.cos(u)
    su = np.sin(u)
    x_orb = a * cu
    y_orb = a * su
    pos = np.zeros((len(times), 3), dtype=float)
    pos[:, 0] = cr * x_orb - sr * ci * y_orb
    pos[:, 1] = sr * x_orb + cr * ci * y_orb
    pos[:, 2] = si * y_orb
    quat = np.asarray(bdata.nadir_attitude([tuple(p) for p in pos], times), dtype=float)
    return times, jd, pos, quat


class Environment:
    """Container for physical satellites and an optional persistent EarthScene."""

    def __init__(self, default_asset=None):
        self.default_asset = default_asset
        self.satellites: list[EnvironmentSatellite] = []
        self.observed_quantities = {}
        self.scene = None

    @property
    def trajectory(self):
        return {sat.sat_id: sat.trajectory_record() for sat in self.satellites}

    def add_satellite(self, t, jd, pos, quat=None, name=None, plane=0, slot=0, asset=None):
        sat_id = len(self.satellites)
        t = np.asarray(t, dtype=float)
        jd = np.asarray(jd, dtype=float)
        pos = np.asarray(pos, dtype=float)
        if quat is None:
            quat = np.asarray(bdata.nadir_attitude([tuple(p) for p in pos], t), dtype=float)
        else:
            quat = np.asarray(quat, dtype=float)
        sat = EnvironmentSatellite(
            sat_id=sat_id,
            plane=int(plane),
            slot=int(slot),
            t=t,
            jd=jd,
            pos=pos,
            quat=quat,
            name=name or f"sat{sat_id}",
            asset=asset or self.default_asset,
        )
        self.satellites.append(sat)
        return sat

    def add_circular_walker(
        self,
        n_planes=3,
        sats_per_plane=6,
        minutes=40.0,
        dt_s=30.0,
        alt_km=550.0,
        inc_deg=53.0,
        phasing=1,
        asset=None,
    ):
        for p in range(int(n_planes)):
            for s in range(int(sats_per_plane)):
                t, jd, pos, quat = circular_orbit_history(
                    p,
                    s,
                    n_planes,
                    sats_per_plane,
                    minutes,
                    dt_s,
                    alt_km=alt_km,
                    inc_deg=inc_deg,
                    phasing=phasing,
                )
                self.add_satellite(t, jd, pos, quat, plane=p, slot=s, asset=asset)
        return self.satellites

    def add_trajectory_csv(self, path, asset=None):
        sats = bdata.load_trajectory(str(path))
        for sid, record in sorted(sats.items()):
            self.add_satellite(
                record["t"],
                record["jd"],
                record["pos"],
                record["quat"],
                name=f"sat{sid}",
                plane=0,
                slot=sid,
                asset=asset,
            )
        return self.satellites

    def add_mission(self, mission, asset=None):
        """Flush a BARREL Mission and import its logged trajectory."""
        mission.flush()
        return self.add_trajectory_csv(Path(mission.run_dir) / "trajectory.csv", asset=asset)

    def add_observed_quantity(self, quantity):
        """Register a truth/sampling object that satellites may observe."""
        self.observed_quantities[quantity.name] = quantity
        return quantity

    def detection_emitter(self, name, rng=None, sample_dt_s=1.0):
        """Expose a registered observed quantity as a scene optical emitter."""
        from .detectable import ObservedQuantityEmitter
        return ObservedQuantityEmitter(self.observed(name), rng=rng, sample_dt_s=sample_dt_s)

    def runtime_scene(self, seed=0, observed_names=(), sample_dt_s=1.0):
        """Create the event-driven view of this environment's recorded ephemeris."""
        from .runtime import Scene
        scene = Scene.from_environment(self, seed=seed)
        for i, name in enumerate(observed_names):
            scene.add_detectable(self.detection_emitter(
                name, rng=np.random.default_rng(int(seed) + 7919 * (i + 1)), sample_dt_s=sample_dt_s
            ))
        return scene

    def observed(self, name):
        return self.observed_quantities[name]

    def sample_observed(self, name, epoch_min, rng, dt_s=1.0):
        return self.observed(name).sample(epoch_min, rng, dt_s=dt_s)

    def truth_frame(self, name, epoch_min):
        return self.observed(name).truth_frame(epoch_min)

    def render_observed(self, scene=None, epoch_min=0.0, names=None):
        scene = scene or self.scene
        if scene is None:
            raise RuntimeError("build_scene() before rendering observed quantities")
        rendered = []
        selected = names or list(self.observed_quantities)
        for name in selected:
            rendered.extend(self.observed_quantities[name].render(scene, epoch_min=epoch_min))
        return rendered

    def write_trajectory_csv(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sat_id", "t_s", "jd_utc", "x_m", "y_m", "z_m", "qx", "qy", "qz", "qw"])
            for sat in self.satellites:
                for k, t in enumerate(sat.t):
                    writer.writerow(
                        [
                            sat.sat_id,
                            f"{float(t):.3f}",
                            f"{float(sat.jd[k]):.9f}",
                            *(f"{float(x):.3f}" for x in sat.pos[k]),
                            *(f"{float(q):.6f}" for q in sat.quat[k]),
                        ]
                    )
        return path

    def build_scene(
        self,
        samples=32,
        resolution=(1600, 900),
        flat=False,
        earth_kwargs=None,
        satellite_asset=None,
        satellite_scale=0.30,
        trails=True,
        trail_width=0.012,
        include_observed=True,
        observed_epoch_min=0.0,
    ):
        """Create one persistent EarthScene and add satellites once.

        Rendering multiple frames through this scene reuses imported meshes and only moves
        objects via EarthScene's frame placement logic.
        """
        from barrel_render import EarthScene

        earth_kwargs = dict(earth_kwargs or {})
        scene = EarthScene(samples=samples, resolution=resolution, flat=flat)
        scene.add_earth(**earth_kwargs)
        chosen_asset = satellite_asset or self.default_asset
        if chosen_asset:
            asset_path(chosen_asset)  # validate early for a clearer error
        scene.add_satellites(
            self.trajectory,
            scale=satellite_scale,
            asset=chosen_asset,
            point_nadir=False,
            trails=trails,
            trail_width=trail_width,
        )
        self.scene = scene
        if include_observed:
            self.render_observed(scene, epoch_min=observed_epoch_min)
        return scene

    def render_still(self, out, frame=-1, camera=None, enabled=True, **scene_kwargs):
        if not enabled:
            return None
        if self.scene is None:
            self.build_scene(**scene_kwargs)
        return self.scene.render_still(str(out), frame=frame, camera=camera)
