"""Walker constellation geometry via the barrel Basilisk Mission.

Builds a Walker-delta shell by looping ``Mission.add_satellite`` over classical elements,
propagates it with barrel's full Basilisk dynamics, and returns the logged ECI ephemeris. The
recorded positions are cached in NumPy format at the precision of barrel's
``trajectory.csv`` so the heavy Basilisk run happens once. Propagation remains barrel's.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

MU_EARTH = 3.986004418e14
R_EARTH = 6378137.0
DEFAULT_ORBIT_EPOCH = "2026-01-01T00:00:00Z"


def ring_geometry(positions, planes, satellites, earth_radius_m=R_EARTH):
    """Measure every adjacent ring segment at every propagated epoch."""
    positions = np.asarray(positions).reshape(-1, planes, satellites, 3)
    if not np.all(np.isfinite(positions)):
        raise ValueError("nonfinite propagated satellite position")
    minimum, maximum_range = np.inf, 0.0
    for slot in range(satellites):
        start = positions[:, :, slot]
        difference = positions[:, :, (slot + 1) % satellites] - start
        distance_squared = np.sum(difference * difference, axis=-1)
        fraction = np.clip(-np.sum(start * difference, axis=-1) / distance_squared, 0, 1)
        nearest = start + fraction[..., None] * difference
        minimum = min(minimum, float(np.min(np.linalg.norm(nearest, axis=-1))))
        maximum_range = max(maximum_range, float(np.sqrt(np.max(distance_squared))))
    return {
        "minimum_limb_clearance_km": (minimum - earth_radius_m) / 1000,
        "maximum_adjacent_range_km": maximum_range / 1000,
        "epochs_checked": positions.shape[0],
        "links_per_epoch": planes * satellites,
        "reference_earth_radius_km": earth_radius_m / 1000,
    }


def _epoch_datetime(epoch):
    if isinstance(epoch, str):
        value = datetime.fromisoformat(epoch.replace("Z", "+00:00"))
    else:
        value = datetime(*epoch, tzinfo=timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _epoch_text(epoch):
    return _epoch_datetime(epoch).strftime("%Y-%m-%dT%H:%M:%SZ")


def _epoch_tuple(epoch):
    value = _epoch_datetime(epoch)
    return value.year, value.month, value.day, value.hour, value.minute, value.second


def _julian_date(epoch):
    return 2440587.5 + _epoch_datetime(epoch).timestamp() / 86400.0


def cache_path(n_planes, sats_per_plane, alt_km, inc_deg, minutes, dt_s, phasing=1,
               raan_offset_deg=0.0, eccentricity=0.001,
               argument_of_perigee_deg=0.0, epoch=DEFAULT_ORBIT_EPOCH, cache_root="cache/sim"):
    raan_tag = "" if abs(float(raan_offset_deg)) < 1e-12 else f"_raan{raan_offset_deg:g}"
    epoch_tag = _epoch_datetime(epoch).strftime("%Y%m%dT%H%M%S")
    tag = (
        f"walker_p{n_planes}_s{sats_per_plane}_f{phasing}_"
        f"h{alt_km:g}_i{inc_deg:g}{raan_tag}_e{eccentricity:g}_"
        f"aop{argument_of_perigee_deg:g}_epoch{epoch_tag}_m{minutes:g}_dt{dt_s:g}"
    )
    return Path(cache_root) / (tag.replace(".", "p") + ".npz")


def metadata_path(cache):
    return cache.with_suffix(".json")


def display_path(cache):
    try:
        return str(cache.relative_to(Path.cwd()))
    except ValueError:
        return str(cache)


def write_metadata(cache, **meta):
    metadata_path(cache).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def walker_elements(n_planes, sats_per_plane, alt_km=550.0, inc_deg=53.0, phasing=1,
                    raan_offset_deg=0.0, eccentricity=0.001,
                    argument_of_perigee_deg=0.0):
    """Classical elements (a,e,i,RAAN,argp,nu) in SI/rad for a Walker-delta shell."""
    from barrel_mission.mission import MU_EARTH, R_EARTH, DEG2RAD   
    a = R_EARTH + alt_km * 1e3
    total = n_planes * sats_per_plane
    els = []
    for p in range(n_planes):
        raan = float(raan_offset_deg) + 360.0 * p / n_planes
        for s in range(sats_per_plane):
            nu = 360.0 * s / sats_per_plane + 360.0 * phasing * p / total
            els.append((
                a,
                float(eccentricity),
                inc_deg * DEG2RAD,
                raan * DEG2RAD,
                float(argument_of_perigee_deg) * DEG2RAD,
                (nu % 360.0) * DEG2RAD,
            ))
    return els


def orbit_period_min(alt_km=550.0):
    a = R_EARTH + float(alt_km) * 1e3
    return 2.0 * math.pi * math.sqrt(a**3 / MU_EARTH) / 60.0


def generate_circular(n_planes=3, sats_per_plane=6, alt_km=550.0, inc_deg=53.0,
                      minutes=40.0, dt_s=30.0, phasing=1, raan_offset_deg=0.0,
                      epoch=DEFAULT_ORBIT_EPOCH):
    """Fast circular Walker-delta ephemeris for parameter sweeps.

    This is intentionally a low-cost test model: circular two-body motion in fixed ECI
    orbital planes, with the same Walker-delta RAAN and phasing convention as the Basilisk
    generator. The returned dictionary matches ``generate`` so the nadir CMOS model and
    tracker do not need a separate code path.
    """
    a = R_EARTH + float(alt_km) * 1e3
    inc = math.radians(float(inc_deg))
    n = math.sqrt(MU_EARTH / a**3)
    total = int(n_planes) * int(sats_per_plane)
    duration_s = float(minutes) * 60.0
    times = np.arange(0.0, duration_s + 0.5 * float(dt_s), float(dt_s))
    jd = _julian_date(epoch) + times / 86400.0
    pos = np.zeros((len(times), total, 3), dtype=float)

    ci, si = math.cos(inc), math.sin(inc)
    sat_index = 0
    for p in range(int(n_planes)):
        raan = math.radians(float(raan_offset_deg)) + 2.0 * math.pi * p / float(n_planes)
        cr, sr = math.cos(raan), math.sin(raan)
        for s in range(int(sats_per_plane)):
            nu0 = 2.0 * math.pi * s / float(sats_per_plane)
            nu0 += 2.0 * math.pi * float(phasing) * p / float(total)
            u = nu0 + n * times
            cu = np.cos(u)
            su = np.sin(u)
            x_orb = a * cu
            y_orb = a * su
            pos[:, sat_index, 0] = cr * x_orb - sr * ci * y_orb
            pos[:, sat_index, 1] = sr * x_orb + cr * ci * y_orb
            pos[:, sat_index, 2] = si * y_orb
            sat_index += 1

    return dict(
        t_s=times,
        jd=jd,
        pos=pos,
        n_sat=total,
        cache_path=(
            f"circular_walker_p{n_planes}_s{sats_per_plane}_f{phasing}_"
            f"h{alt_km:g}_i{inc_deg:g}_raan{raan_offset_deg:g}"
        ),
        propagator="circular two-body Walker test model",
    )


def generate(n_planes=3, sats_per_plane=6, alt_km=550.0, inc_deg=53.0,
             minutes=40.0, dt_s=30.0, phasing=1, force=False, raan_offset_deg=0.0,
             eccentricity=0.001, argument_of_perigee_deg=0.0,
             epoch=DEFAULT_ORBIT_EPOCH, workers=1, cache_root="cache/sim"):
    """Propagate a Walker shell with barrel/Basilisk and return the ECI ephemeris.

    Returns dict with: t_s (T,), jd (T,), pos (T, N, 3) ECI metres, n_sat.
    """
    epoch_text = _epoch_text(epoch)
    cache = cache_path(
        n_planes, sats_per_plane, alt_km, inc_deg, minutes, dt_s, phasing,
        raan_offset_deg, eccentricity, argument_of_perigee_deg, epoch_text, cache_root)
    if cache.exists() and not force:
        d = np.load(cache)
        def meta(name, default):
            return d[name] if name in d.files else np.array(default)
        expected = n_planes * sats_per_plane
        if (
            int(d["n_sat"]) == expected
            and "minutes" in d.files
            and "dt_s" in d.files
            and int(meta("n_planes", n_planes)) == n_planes
            and int(meta("sats_per_plane", sats_per_plane)) == sats_per_plane
            and int(meta("phasing", phasing)) == phasing
            and abs(float(meta("alt_km", alt_km)) - alt_km) < 1e-9
            and abs(float(meta("inc_deg", inc_deg)) - inc_deg) < 1e-9
            and abs(float(meta("raan_offset_deg", raan_offset_deg)) - raan_offset_deg) < 1e-9
            and abs(float(meta("eccentricity", np.nan)) - eccentricity) < 1e-12
            and abs(float(meta("argument_of_perigee_deg", np.nan))
                    - argument_of_perigee_deg) < 1e-9
            and str(meta("orbit_epoch", "")) == epoch_text
            and abs(float(meta("minutes", minutes)) - minutes) < 1e-9
            and abs(float(meta("dt_s", dt_s)) - dt_s) < 1e-9
        ):
            write_metadata(
                cache,
                cache_file=display_path(cache),
                n_sat=int(d["n_sat"]),
                n_planes=int(n_planes),
                sats_per_plane=int(sats_per_plane),
                phasing=int(phasing),
                alt_km=float(alt_km),
                inc_deg=float(inc_deg),
                raan_offset_deg=float(raan_offset_deg),
                eccentricity=float(eccentricity),
                argument_of_perigee_deg=float(argument_of_perigee_deg),
                orbit_epoch=epoch_text,
                minutes=float(minutes),
                dt_s=float(dt_s),
                n_epochs=int(len(d["t_s"])),
                pos_shape=list(d["pos"].shape),
                position_frame="ECI",
                position_units="m",
                time_units="s",
                propagator="Basilisk via barrel Mission",
            )
            return dict(
                t_s=d["t_s"],
                jd=d["jd"],
                pos=d["pos"],
                n_sat=int(d["n_sat"]),
                cache_path=display_path(cache),
            )

    elements = walker_elements(
        n_planes, sats_per_plane, alt_km, inc_deg, phasing, raan_offset_deg,
        eccentricity, argument_of_perigee_deg,
    )
    groups = np.array_split(np.arange(len(elements)), min(int(workers), len(elements)))
    jobs = [(f"comms_{cache.stem}_part{index}", _epoch_tuple(epoch_text),
             [elements[item] for item in group], minutes, dt_s, str(Path(cache_root) / "missions"))
            for index, group in enumerate(groups)]
    if len(jobs) == 1:
        pieces = [_propagate_group(jobs[0])]
    else:
        with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
            pieces = list(pool.map(_propagate_group, jobs))
    times, jd = pieces[0][:2]
    for other_times, other_jd, _ in pieces[1:]:
        np.testing.assert_array_equal(other_times, times)
        np.testing.assert_array_equal(other_jd, jd)
    pos = np.concatenate([piece[2] for piece in pieces], axis=1)
    n_sat = pos.shape[1]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        t_s=times,
        jd=jd,
        pos=pos,
        n_sat=n_sat,
        n_planes=n_planes,
        sats_per_plane=sats_per_plane,
        phasing=phasing,
        alt_km=alt_km,
        inc_deg=inc_deg,
        raan_offset_deg=raan_offset_deg,
        eccentricity=eccentricity,
        argument_of_perigee_deg=argument_of_perigee_deg,
        orbit_epoch=np.asarray(epoch_text),
        minutes=minutes,
        dt_s=dt_s,
    )
    write_metadata(
        cache,
        cache_file=display_path(cache),
        n_sat=int(n_sat),
        n_planes=int(n_planes),
        sats_per_plane=int(sats_per_plane),
        phasing=int(phasing),
        alt_km=float(alt_km),
        inc_deg=float(inc_deg),
        raan_offset_deg=float(raan_offset_deg),
        eccentricity=float(eccentricity),
        argument_of_perigee_deg=float(argument_of_perigee_deg),
        orbit_epoch=epoch_text,
        minutes=float(minutes),
        dt_s=float(dt_s),
        n_epochs=int(len(times)),
        pos_shape=list(pos.shape),
        position_frame="ECI",
        position_units="m",
        time_units="s",
        propagator="Basilisk via barrel Mission",
    )
    return dict(t_s=times, jd=jd, pos=pos, n_sat=n_sat, cache_path=display_path(cache))


def _propagate_group(arguments):
    """Spacecraft dynamics are independent, so groups can propagate separately."""
    from barrel_mission import Mission

    name, epoch, elements, minutes, dt_s, output_root = arguments
    mission = Mission(name=name, epoch=epoch, output_root=output_root)
    for index, element in enumerate(elements):
        mission.add_satellite(elements=element, name=f"sat{index}",
                              pointing="nadir", cameras=[("overview",)])
    mission.run(duration=minutes * 60.0, dt=dt_s)
    result = recorded_ephemeris(mission)
    mission.close()
    return result


def recorded_ephemeris(mission):
    """Read Basilisk logs at the precision of Mission's trajectory CSV.

    Geometry needs position and time only. Avoid exporting and parsing millions
    of orbital-element and attitude rows for a constellation parameter sweep.
    """
    times = np.round(np.asarray(mission._recs[0].times()) * 1e-9, 3)
    position = np.stack([np.asarray(record.r_BN_N) for record in mission._recs], axis=1)
    invalid = np.flatnonzero(~np.all(np.isfinite(position), axis=(0, 2)))
    if len(invalid):
        raise ValueError(f"nonfinite Basilisk positions for satellite indices {invalid.tolist()}")
    jd = np.fromiter((float(f"{value:.9f}") for value in mission.jd0 + times / 86400.0),
                     dtype=float, count=len(times))
    return times, jd, np.round(position, 3)



@dataclass(frozen=True)
class WalkerDeltaConstellation:
    """One Walker-delta shell, with reusable elements and ephemeris generation."""

    n_planes: int = 3
    sats_per_plane: int = 6
    alt_km: float = 550.0
    inc_deg: float = 53.0
    phasing: int = 1
    raan_offset_deg: float = 0.0
    eccentricity: float = 0.001
    argument_of_perigee_deg: float = 0.0

    @property
    def satellite_count(self):
        return self.n_planes * self.sats_per_plane

    def elements(self):
        return walker_elements(**asdict(self))

    def propagate(self, minutes=40.0, dt_s=30.0, *, epoch=DEFAULT_ORBIT_EPOCH,
                  cache_root="cache/sim", force=False, workers=1):
        return generate(**asdict(self), minutes=minutes, dt_s=dt_s, epoch=epoch,
                        cache_root=cache_root, force=force, workers=workers)

    def circular(self, minutes=40.0, dt_s=30.0, *, epoch=DEFAULT_ORBIT_EPOCH):
        parameters = asdict(self)
        parameters.pop("eccentricity")
        parameters.pop("argument_of_perigee_deg")
        return generate_circular(**parameters, minutes=minutes, dt_s=dt_s, epoch=epoch)

    def ring_geometry(self, positions):
        return ring_geometry(positions, self.n_planes, self.sats_per_plane)
