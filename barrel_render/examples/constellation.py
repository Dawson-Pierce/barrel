"""Example 2 -- a constellation, and what each satellite observes.

Simulates several satellites and renders:
  * output/constellation.png         -- a wide overview of the whole constellation
  * output/views/sat<N>.png          -- one off-nadir EO frame per satellite (what it sees)
  * output/views/stereo_0_L/R.png    -- a body-mounted stereo rig on satellite 0

This shows the multi-camera framework: many cameras live in one scene at once
(a wide overview, one off-nadir sensor per satellite tilted toward the horizon so
the curved Earth + limb fill the frame, plus a body-mounted stereo pair), and
render_views() renders the scene from each in turn. The stereo rig is parented to
the satellite, so the attitude dynamics drive it.

    python barrel_render/examples/constellation.py

Same two-mode structure as follow_satellite.py: the host branch drives the C++
sim + Docker; the Blender branch builds the scene with EarthScene.
"""

import os
import sys

# repo root (contains the barrel_render package) = three levels up
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import bpy  # noqa: F401  (only present inside Blender)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False

# --------------------------------------------------------------------------- #
# SETTINGS
N_SATELLITES = 6
STEP_SECONDS = "10"
OVERVIEW_RES = (1600, 900)
VIEW_RES = (640, 640)        # per-satellite EO frame (square sensor)
SAMPLES = 48
SENSOR_FOV_DEG = 30.0        # field of view of each satellite's camera
SENSOR_OFF_NADIR_DEG = 55.0  # tilt forward of nadir so the curved Earth + limb fill the frame
STEREO_BASELINE_M = 3.0      # body-mounted stereo separation (metres)
STEREO_CONVERGE_DEG = 6.0    # toe-in so the L/R frames visibly differ
EARTH_RES = "8k"             # which Texture_Files/Earth-<res>/ maps to use: 8k | 16k | 32k
EARTH_GLB = "glb_files/Earth.glb"   # fallback model if Texture_Files maps absent
# --------------------------------------------------------------------------- #


def build_scene(data_csv, overview_out, views_pattern, earth_glb=None,
                color="", bump="", normal="", night="", specular="",
                stars="", moon="", sun=""):
    """Inside Blender: build the constellation and render the overview + EO views."""
    from barrel_render import EarthScene, data

    sats = data.load_trajectory(data_csv)
    jd0 = next(iter(sats.values()))["jd"][0]
    sun_dir = data.sun_direction_eci(jd0)

    scene = EarthScene(samples=SAMPLES, resolution=OVERVIEW_RES)
    scene.add_stars(texture=stars or None)               # star-map sky (inertial)
    # Photoreal Earth from Texture_Files maps if present; else glb; else procedural.
    scene.add_earth(color_map=color or None, bump_map=bump or None, normal_map=normal or None,
                    night_map=night or None, specular_map=specular or None,
                    glb=earth_glb or None, sun_direction=sun_dir)
    scene.add_atmosphere(sun_direction=sun_dir)
    scene.add_sun(jd=jd0, direction=sun_dir, texture=sun or None)
    scene.add_moon(jd=jd0, texture=moon or None)
    # Nadir-pointing so the body frame is meaningful and the mounted cameras
    # below image the ground (this orbit-only sim has no real attitude).
    scene.add_satellites(sats, point_nadir=True)

    # Many cameras, one scene: a wide overview + one nadir sensor per satellite +
    # a body-mounted stereo rig on satellite 0 that rides its attitude dynamics.
    scene.camera_overview()                              # active camera for the overview
    sensor_cams = scene.add_sensor_cameras(fov_deg=SENSOR_FOV_DEG,
                                           off_nadir_deg=SENSOR_OFF_NADIR_DEG)
    stereo = scene.add_stereo_pair(sat_id=0, baseline=STEREO_BASELINE_M, axis="y",
                                   fov_deg=SENSOR_FOV_DEG, converge_deg=STEREO_CONVERGE_DEG,
                                   pitch_deg=SENSOR_OFF_NADIR_DEG)

    # 1) the wide shot of the whole constellation
    scene.render_still(overview_out, camera="overview")

    # 2) what each satellite observes + the stereo pair (square frames, lower res)
    bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y = VIEW_RES
    scene.render_views(views_pattern, names=list(sensor_cams) + list(stereo))


def host():
    """On the host: simulate -> render overview + per-satellite views in Blender."""
    import _pipeline as P

    csv = P.simulate(N_SATELLITES, "constellation.csv", step=STEP_SECONDS)
    P.fresh_dir("/work/output/views")
    m = P.earth_maps(EARTH_RES)
    P.render(__file__, ["--data", csv,
                        "--overview", "/work/output/constellation.png",
                        "--views", "/work/output/views/{name}.png",
                        "--earth-glb", P.asset(EARTH_GLB),
                        "--earth-color", m["color"], "--earth-bump", m["bump"],
                        "--earth-normal", m["normal"], "--earth-night", m["night"],
                        "--earth-specular", m["specular"],
                        "--stars", P.body_texture("stars"),
                        "--moon", P.body_texture("moon"),
                        "--sun", P.body_texture("sun")])
    print("\nDone:")
    print("  output/constellation.png    (overview of all satellites)")
    print("  output/views/sat*.png       (what each satellite observes)")
    print("  output/views/stereo_0_*.png (body-mounted stereo rig on sat 0)")


if _IN_BLENDER:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--overview", required=True)
    p.add_argument("--views", required=True)
    p.add_argument("--earth-glb", default="")
    p.add_argument("--earth-color", default="")
    p.add_argument("--earth-bump", default="")
    p.add_argument("--earth-normal", default="")
    p.add_argument("--earth-night", default="")
    p.add_argument("--earth-specular", default="")
    p.add_argument("--stars", default="")
    p.add_argument("--moon", default="")
    p.add_argument("--sun", default="")
    a = p.parse_args(argv)
    build_scene(a.data, a.overview, a.views, earth_glb=a.earth_glb,
                color=a.earth_color, bump=a.earth_bump, normal=a.earth_normal,
                night=a.earth_night, specular=a.earth_specular,
                stars=a.stars, moon=a.moon, sun=a.sun)
elif __name__ == "__main__":
    host()
