"""Example 1 -- a single satellite with the camera chasing it.

Simulates one satellite, then renders an orbital chase-camera animation and
stitches it into a looping GIF.

    python barrel_render/examples/follow_satellite.py
    -> output/orbit.gif

The file runs in two modes: on the host it drives the C++ sim + Docker + ffmpeg;
inside Blender (it re-invokes itself there) it builds the scene with EarthScene.
Tweak the SETTINGS block below for resolution / length / quality.
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
N_SATELLITES = 1
STEP_SECONDS = "10"          # C++ integration / sample step
RESOLUTION = (960, 540)
SAMPLES = 24                 # Cycles samples (higher = cleaner, slower)
MAX_FRAMES = 60              # subsample the trajectory to this many frames
FPS = 20
GIF_WIDTH = 720
EARTH_RES = "8k"             # which Texture_Files/Earth-<res>/ maps to use: 8k | 16k | 32k
EARTH_GLB = "glb_files/Earth.glb"   # fallback model if Texture_Files maps absent
# --------------------------------------------------------------------------- #


def build_scene(data_csv, out_dir, earth_glb=None, color="", bump="", normal="",
                night="", specular="", stars="", moon="", sun=""):
    """Inside Blender: build the scene and render the chase-camera animation."""
    from barrel_render import EarthScene, data

    sats = data.load_trajectory(data_csv)
    jd0 = next(iter(sats.values()))["jd"][0]
    sun_dir = data.sun_direction_eci(jd0)

    scene = EarthScene(samples=SAMPLES, resolution=RESOLUTION)
    scene.add_stars(texture=stars or None)               # star-map sky (inertial)
    # Photoreal Earth from Texture_Files maps if present; else glb; else procedural.
    scene.add_earth(color_map=color or None, bump_map=bump or None, normal_map=normal or None,
                    night_map=night or None, specular_map=specular or None,
                    glb=earth_glb or None, sun_direction=sun_dir)
    scene.add_atmosphere(sun_direction=sun_dir)
    scene.add_sun(jd=jd0, direction=sun_dir, texture=sun or None)
    scene.add_moon(jd=jd0, texture=moon or None)
    scene.add_satellites(sats, point_nadir=True)   # nadir-pointing bus
    scene.camera_follow(sat_id=0)              # orbital chase camera
    scene.render_animation(out_dir + "/frame_", fps=FPS, max_frames=MAX_FRAMES)


def host():
    """On the host: simulate -> render in Blender (Docker) -> assemble GIF."""
    import _pipeline as P
    from barrel_render import data

    csv = P.simulate(N_SATELLITES, "one_sat.csv", step=STEP_SECONDS)
    frames = "/work/output/frames"
    P.fresh_dir(frames)
    m = P.earth_maps(EARTH_RES)
    P.render(__file__, ["--data", csv, "--out", frames,
                        "--res", RESOLUTION[0], RESOLUTION[1],
                        "--earth-glb", P.asset(EARTH_GLB),
                        "--earth-color", m["color"], "--earth-bump", m["bump"],
                        "--earth-normal", m["normal"], "--earth-night", m["night"],
                        "--earth-specular", m["specular"],
                        "--stars", P.body_texture("stars"),
                        "--moon", P.body_texture("moon"),
                        "--sun", P.body_texture("sun")])
    # mission clock: the trajectory spans exactly one orbital period
    times = data.load_trajectory(os.path.join(P.OUT, "one_sat.csv"))[0]["t"]
    n = len(times)
    n_frames = MAX_FRAMES if (MAX_FRAMES and 1 < MAX_FRAMES < n) else n
    P.make_gif(frames + "/frame_%04d.png", "/work/output/orbit.gif",
               fps=FPS, width=GIF_WIDTH, period_s=times[-1], n_frames=n_frames)
    print(f"\nDone -> output/orbit.gif   (1 orbit ~ {times[-1] / 60:.1f} min)")


if _IN_BLENDER:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--res", type=int, nargs=2, default=list(RESOLUTION))
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
    RESOLUTION = tuple(a.res)
    build_scene(a.data, a.out, earth_glb=a.earth_glb, color=a.earth_color,
                bump=a.earth_bump, normal=a.earth_normal, night=a.earth_night,
                specular=a.earth_specular, stars=a.stars, moon=a.moon, sun=a.sun)
elif __name__ == "__main__":
    host()
