"""Render the Walker-Delta handoff topology with BARREL/Blender.

Host mode exports the existing BARREL/Basilisk Walker ephemeris cache into the BARREL
trajectory CSV contract, then launches this file inside Blender to render:

  output/walker_delta_handoff_blender.png
"""

import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import bpy  # noqa: F401
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False

N_PLANES = 4
SATS_PER_PLANE = 12
GROUND_STATION_LONLAT = (-75.47, 37.94)
OVERVIEW_RES = (1800, 1100)
SAMPLES = 64
EARTH_RES = "8k"
R_EARTH_M = 6378137.0
LINK_MARGIN_M = 80e3
HIGHLIGHT_PLANE = 1


def ecef_to_eci(p_ecef, jd):
    from barrel_render import data
    g = data.gmst(jd)
    c, s = math.cos(g), math.sin(g)
    x, y, z = p_ecef
    return (c * x - s * y, s * x + c * y, z)


def ground_station_eci(lonlat, jd):
    from barrel_render import data
    lon, lat = math.radians(lonlat[0]), math.radians(lonlat[1])
    return ecef_to_eci(data.geodetic_to_ecef(lat, lon, 0.0), jd)


def ring_order_to_downlink(downlink_slot, sats_per_plane, direction=-1):
    return [int((downlink_slot - direction * (sats_per_plane - 1 - k)) % sats_per_plane)
            for k in range(sats_per_plane)]


def closest_downlink_slots(sats, n_planes, sats_per_plane, ground_station):
    import numpy as np
    jd = next(iter(sats.values()))["jd"][0]
    gs = np.asarray(ground_station_eci(ground_station, jd), dtype=float)
    slots = []
    for p in range(n_planes):
        visible = []
        fallback = []
        for s in range(sats_per_plane):
            sid = p * sats_per_plane + s
            pos = np.asarray(sats[sid]["pos"][0], dtype=float)
            d = float(np.linalg.norm(pos - gs))
            fallback.append((d, s))
            if ground_visible(pos, gs):
                visible.append((d, s))
        slots.append(int(min(visible or fallback)[1]))
    return slots


def segment_min_radius(a, b):
    ax, ay, az = a
    bx, by, bz = b
    vx, vy, vz = bx - ax, by - ay, bz - az
    vv = vx * vx + vy * vy + vz * vz
    if vv <= 0.0:
        return math.sqrt(ax * ax + ay * ay + az * az)
    t = -((ax * vx + ay * vy + az * vz) / vv)
    t = min(1.0, max(0.0, t))
    x, y, z = ax + t * vx, ay + t * vy, az + t * vz
    return math.sqrt(x * x + y * y + z * z)


def has_clear_los(a, b, margin_m=LINK_MARGIN_M):
    return segment_min_radius(a, b) >= R_EARTH_M + float(margin_m)


def ground_visible(sat_pos, gs_pos):
    gx, gy, gz = gs_pos
    sx, sy, sz = sat_pos
    ux, uy, uz = gx / R_EARTH_M, gy / R_EARTH_M, gz / R_EARTH_M
    return (sx - gx) * ux + (sy - gy) * uy + (sz - gz) * uz > 0.0


def export_walker_csv(out_csv, n_planes=N_PLANES, sats_per_plane=SATS_PER_PLANE):
    pierced = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.insert(0, pierced)
    from sim import geometry

    g = geometry.generate(n_planes=n_planes, sats_per_plane=sats_per_plane,
                          minutes=40.0, dt_s=30.0, force=False)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "jd_utc", "sat_id", "x_m", "y_m", "z_m"])
        for k, t in enumerate(g["t_s"]):
            for sid in range(int(g["n_sat"])):
                x, y, z = g["pos"][k, sid]
                w.writerow([float(t), float(g["jd"][k]), sid, float(x), float(y), float(z)])
    return "/work/output/" + os.path.basename(out_csv)


def build_scene(data_csv, out_png, earth_glb=None, color="", bump="", normal="",
                night="", specular="", stars="", moon="", sun="", satellite_asset="cygnss"):
    from barrel_render import EarthScene, data

    sats = data.load_trajectory(data_csv)
    jd0 = next(iter(sats.values()))["jd"][0]
    sun_dir = data.sun_direction_eci(jd0)
    gs = ground_station_eci(GROUND_STATION_LONLAT, jd0)
    downlink_slots = closest_downlink_slots(sats, N_PLANES, SATS_PER_PLANE, GROUND_STATION_LONLAT)
    link_color = (0.72, 0.95, 1.0)

    scene = EarthScene(samples=SAMPLES, resolution=OVERVIEW_RES, onboard_exposure=-1.0)
    scene.add_stars(texture=stars or None)
    scene.add_earth(color_map=color or None, bump_map=bump or None, normal_map=normal or None,
                    night_map=night or None, specular_map=specular or None,
                    glb=earth_glb or None, sun_direction=sun_dir)
    scene.add_atmosphere(sun_direction=sun_dir)
    scene.add_sun(jd=jd0, direction=sun_dir, texture=sun or None)
    scene.add_moon(jd=jd0, texture=moon or None)
    scene.add_satellites(
        sats,
        scale=0.16,
        asset=satellite_asset or None,
        point_nadir=True,
        trails=True,
        trail_width=0.004,
    )

    p = HIGHLIGHT_PLANE
    first = p * SATS_PER_PLANE
    dslot = downlink_slots[p]
    order = ring_order_to_downlink(dslot, SATS_PER_PLANE, direction=-1)
    for a, b in zip(order[:-1], order[1:]):
        pa = sats[first + a]["pos"][0]
        pb = sats[first + b]["pos"][0]
        if has_clear_los(pa, pb):
            scene.add_line([pa, pb], frame="eci", color=link_color, width=0.01, strength=5.5)
    highlight_points = [sats[first + s]["pos"][0] for s in range(SATS_PER_PLANE)]
    scene.add_scatter(highlight_points, frame="eci", color=(0.72, 0.95, 1.0),
                      size=0.045, strength=4.0)
    dsid = first + dslot
    scene.add_scatter([sats[dsid]["pos"][0]], frame="eci", color=(1.0, 0.48, 0.05),
                      size=0.075, strength=5.0)
    if ground_visible(sats[dsid]["pos"][0], gs):
        scene.add_line([sats[dsid]["pos"][0], gs], frame="eci", color=(1.0, 0.48, 0.05),
                       width=0.012, strength=6.0)

    scene.add_scatter([gs], frame="eci", color=(1.0, 0.48, 0.05), size=0.06, strength=5.0)
    scene.camera_overview(lens=28.0)
    scene.render_still(out_png, camera="overview")


def host():
    import _pipeline as P
    local_csv = os.path.join(P.OUT, "walker_delta_handoff.csv")
    csv_path = export_walker_csv(local_csv)
    m = P.earth_maps(EARTH_RES)
    P.render(__file__, [
        "--data", csv_path,
        "--out", "/work/output/walker_delta_handoff_blender.png",
        "--earth-color", m["color"],
        "--earth-bump", m["bump"],
        "--earth-normal", m["normal"],
        "--earth-night", m["night"],
        "--earth-specular", m["specular"],
        "--stars", P.body_texture("stars"),
        "--moon", P.body_texture("moon"),
        "--sun", P.body_texture("sun"),
        "--satellite-asset", "cygnss",
    ])
    print("Done: output/walker_delta_handoff_blender.png")


if _IN_BLENDER:
    import argparse
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--earth-glb", default="")
    p.add_argument("--earth-color", default="")
    p.add_argument("--earth-bump", default="")
    p.add_argument("--earth-normal", default="")
    p.add_argument("--earth-night", default="")
    p.add_argument("--earth-specular", default="")
    p.add_argument("--stars", default="")
    p.add_argument("--moon", default="")
    p.add_argument("--sun", default="")
    p.add_argument("--satellite-asset", default="cygnss")
    a = p.parse_args(argv)
    build_scene(a.data, a.out, earth_glb=a.earth_glb, color=a.earth_color,
                bump=a.earth_bump, normal=a.earth_normal, night=a.earth_night,
                specular=a.earth_specular, stars=a.stars, moon=a.moon, sun=a.sun,
                satellite_asset=a.satellite_asset)
elif __name__ == "__main__":
    host()
