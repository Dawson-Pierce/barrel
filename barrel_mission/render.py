"""CSV-driven render pass for a Mission run.

Host half (`render_run`): resolve frame indices + textures, then invoke Blender
ONCE on this file. Blender half: rebuild the scene from the run's trajectory CSV
and render a zoomed-out constellation overview plus each satellite's own cameras
for the requested frames -- not one Blender launch per timestep.
"""

import os
import shutil
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

try:
    import bpy  # noqa: F401
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False

SAMPLES = 32


def render_run(mission, cadence="all", cameras="per_sat", clouds=None, flat=False,
               overview_res=(1600, 900), view_res=(640, 640), earth_res=None,
               satellite_asset=None):
    """Render frames from `mission`'s CSV. cadence = "all" | N | list of indices.
    `clouds` = path to an equirectangular cloud/IR coverage image to overlay.
    `flat=True` = the flat-Earth Easter egg (azimuthal disk; satellites circle it)."""
    import json
    from barrel_render import pipeline as P

    n = mission.frames_logged
    if n == 0:
        raise RuntimeError("nothing logged yet; call step()/run() before render()")
    if cadence == "all":
        idx = list(range(n))
    elif isinstance(cadence, int):
        idx = list(range(0, n, max(1, cadence)))
    else:
        idx = [int(i) for i in cadence]

    res = earth_res or mission.earth_res
    m = P.earth_maps(res)
    csv = os.path.join(mission.run_dir, "trajectory.csv")
    outdir = os.path.join(mission.run_dir, "renders")
    shutil.rmtree(outdir, ignore_errors=True)
    os.makedirs(outdir, exist_ok=True)
    specs = [[s["name"], s["cameras"]] for s in mission._sats]

    args = ["--data", csv, "--out-dir", outdir,
            "--frames", ",".join(str(i) for i in idx),
            "--cameras-json", json.dumps(specs), "--cameras-mode", cameras,
            "--ov-res", overview_res[0], overview_res[1],
            "--view-res", view_res[0], view_res[1],
            "--earth-color", m["color"], "--earth-bump", m["bump"],
            "--earth-normal", m["normal"], "--earth-night", m["night"],
            "--earth-specular", m["specular"], "--stars", P.body_texture("stars"),
            "--moon", P.body_texture("moon"), "--sun", P.body_texture("sun"),
            "--clouds", clouds or P.body_texture("clouds"), "--flat", "1" if flat else "",
            "--satellite-asset", satellite_asset or ""]
    P.render(__file__, args)
    print(f"[mission] renders -> {outdir}  ({len(idx)} frame(s))")
    return outdir


if _IN_BLENDER:
    import argparse
    import json

    from barrel_render import EarthScene, data

    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--frames", default="")
    p.add_argument("--cameras-json", default="[]")
    p.add_argument("--cameras-mode", default="per_sat")
    p.add_argument("--ov-res", type=int, nargs=2, default=[1600, 900])
    p.add_argument("--view-res", type=int, nargs=2, default=[640, 640])
    for opt in ("color", "bump", "normal", "night", "specular"):
        p.add_argument(f"--earth-{opt}", default="")
    p.add_argument("--stars", default="")
    p.add_argument("--moon", default="")
    p.add_argument("--sun", default="")
    p.add_argument("--clouds", default="")
    p.add_argument("--flat", default="")
    p.add_argument("--satellite-asset", default="")
    a = p.parse_args(argv)
    flat = bool(a.flat)

    sats = data.load_trajectory(a.data)
    jd0 = next(iter(sats.values()))["jd"][0]
    sun_dir = data.sun_direction_eci(jd0)
    frames = [int(x) for x in a.frames.split(",") if x.strip() != ""]

    # Clouds: a single image (constant) or a printf pattern like ".../ir_%04d.png"
    # where %d is the frame index (time-evolving clouds from IR time-series).
    cloud_seq = (not flat) and "%" in a.clouds
    def cloud_for(fi):
        return (a.clouds % fi) if cloud_seq else a.clouds

    scene = EarthScene(samples=SAMPLES, resolution=tuple(a.ov_res), flat=flat)
    scene.add_stars(texture=a.stars or None)
    if flat:                                             # flat-Earth Easter egg
        sun_ecef = data.eci_to_ecef(sun_dir, jd0)        # globe-matched day/night
        scene.add_earth(color_map=a.earth_color or None, night_map=a.earth_night or None,
                        flat_sun_ecef=sun_ecef)
        scene.add_sun(direction=(0.0, 0.0, 1.0))         # light the satellites from above
    else:
        scene.add_earth(color_map=a.earth_color or None, bump_map=a.earth_bump or None,
                        normal_map=a.earth_normal or None, night_map=a.earth_night or None,
                        specular_map=a.earth_specular or None, sun_direction=sun_dir)
        if a.clouds:
            scene.add_clouds(cloud_for(frames[0]) if frames else a.clouds)
        scene.add_atmosphere(sun_direction=sun_dir)
        scene.add_sun(jd=jd0, direction=sun_dir, texture=a.sun or None)
        scene.add_moon(jd=jd0, texture=a.moon or None)
    scene.add_satellites(sats, asset=a.satellite_asset or None, point_nadir=not flat)
    if flat:                                             # placed above the satellites
        scene.add_flat_sun(sun_ecef, texture=a.sun or None)          # low Sun, sub-solar pt
        moon_ecef = data.eci_to_ecef(data.moon_position_eci(jd0), jd0)
        scene.add_flat_moon(moon_ecef, texture=a.moon or None)       # low Moon, sub-lunar pt
    scene.camera_overview()                              # auto-frames everything (zoomed out)

    sensor_names = []
    if a.cameras_mode == "per_sat" and not flat:         # onboard EO views: globe only
        for sid, (name, cams) in enumerate(json.loads(a.cameras_json)):
            for spec in cams:
                if spec and spec[0] == "sensor":
                    fov = float(spec[1]) if len(spec) > 1 else 30.0
                    off = float(spec[2]) if len(spec) > 2 else 55.0
                    scene.camera_sensor(sat_id=sid, fov_deg=fov, off_nadir_deg=off,
                                        name=name, make_active=False)
                    sensor_names.append(name)

    for fi in frames:
        if cloud_seq:
            scene.set_cloud_image(cloud_for(fi))         # time-evolving clouds
        bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y = tuple(a.ov_res)
        scene.render_still(f"{a.out_dir}/overview_{fi:04d}.png", frame=fi, camera="overview")
        if sensor_names:
            bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y = tuple(a.view_res)
            scene.render_views(a.out_dir + "/{name}_" + f"{fi:04d}.png", frame=fi,
                               names=sensor_names)
    print(f"MISSION render done: {len(frames)} frame(s), {len(sensor_names)} sensor cam(s)")
