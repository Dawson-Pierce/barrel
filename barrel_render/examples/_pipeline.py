"""Host-side plumbing shared by the examples (no bpy): propagate with the Basilisk
Mission, build the Blender image if needed, render an example file inside Blender
(Docker), and stitch frames into a GIF. Internal helper -- see the example scripts.
"""

import glob
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "output")
IMAGE = "barrel-blender"
MOUNT = REPO.replace("\\", "/") + ":/work"


def _run(cmd, **kw):
    print("    $ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def simulate(n_sats, out_name, step="10"):
    """Propagate a Walker-like LEO constellation with the Basilisk Mission and write
    output/<out_name> (a trajectory CSV); returns the '/work/...' container path."""
    import math
    sys.path.insert(0, REPO)
    from barrel_mission.mission import Mission, MU_EARTH, R_EARTH, DEG2RAD

    os.makedirs(OUT, exist_ok=True)
    stem = os.path.splitext(out_name)[0]
    a = R_EARTH + 550e3
    period = 2.0 * math.pi * math.sqrt(a ** 3 / MU_EARTH)
    m = Mission(name=stem, output_root=OUT, forces=("j2",))
    for k in range(int(n_sats)):
        m.add_satellite(
            elements=(a, 0.001, 53 * DEG2RAD, (360.0 * k / n_sats) * DEG2RAD, 0.0,
                      (360.0 * ((k * 7) % n_sats) / n_sats) * DEG2RAD),
            name=f"sat{k}", pointing="nadir")
    m.run(duration=period, dt=float(step))
    m.close()
    shutil.copyfile(os.path.join(m.run_dir, "trajectory.csv"), os.path.join(OUT, out_name))
    return "/work/output/" + out_name


def ensure_image():
    if shutil.which("docker") is None:
        sys.exit("Docker is required for rendering. Start Docker Desktop and retry.")
    have = subprocess.run(["docker", "image", "inspect", IMAGE],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    if not have:
        print("    (building the BARREL image once; downloads Blender ~300 MB + Basilisk)")
        _run(["docker", "build", "-t", IMAGE, "."], cwd=REPO)


_IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr")


def asset(rel):
    """Container path ('/work/...') for a repo-relative asset if it exists, else ''."""
    return "/work/" + rel.replace("\\", "/") if os.path.exists(os.path.join(REPO, rel)) else ""


def _first_image(absdir):
    """'/work/...' path of the first image file in a directory, else ''."""
    if os.path.isdir(absdir):
        for fn in sorted(os.listdir(absdir)):
            if fn.lower().endswith(_IMG_EXTS):
                rel = os.path.relpath(os.path.join(absdir, fn), REPO).replace("\\", "/")
                return "/work/" + rel
    return ""


def earth_maps(res="8k", folder="Texture_Files"):
    """Resolve Earth maps in Texture_Files/Earth-<res>/ by filename keyword.
    Returns '/work/...' container paths for color/bump/normal/night/specular
    ('' if absent)."""
    out = {"color": "", "bump": "", "normal": "", "night": "", "specular": ""}
    d = os.path.join(REPO, folder, "Earth-" + res)
    if not os.path.isdir(d):
        return out
    rel = os.path.relpath(d, REPO).replace("\\", "/")
    keys = {"color": ("color", "day", "albedo", "diffuse"),
            "normal": ("normal", "norm"),
            "bump": ("bump", "topo", "elev", "height", "displace"),
            "night": ("night", "lights"),
            "specular": ("specular", "spec", "gloss", "water")}
    for fn in sorted(os.listdir(d)):
        low = fn.lower()
        if not low.endswith(_IMG_EXTS):
            continue
        for key, words in keys.items():
            if not out[key] and any(w in low for w in words):
                out[key] = "/work/" + rel + "/" + fn
                break
    return out


def body_texture(kind, folder="Texture_Files"):
    """First image in Texture_Files/<kind>-*/ (e.g. moon, stars, sun), '' if none."""
    base = os.path.join(REPO, folder)
    if os.path.isdir(base):
        for sub in sorted(os.listdir(base)):
            if sub.lower().startswith(kind.lower() + "-"):
                img = _first_image(os.path.join(base, sub))
                if img:
                    return img
    return ""


def render(example_file, args):
    """Render `example_file` inside Blender. If a local `blender` is on PATH (i.e.
    we're already inside the barrel-blender image), call it directly -- the whole
    pipeline runs in one container. Otherwise (host) render via `docker run`."""
    rel = os.path.relpath(os.path.abspath(example_file), REPO).replace("\\", "/")
    local_blender = shutil.which("blender")
    tail = [str(a) for a in args]
    if local_blender:  # in-container: single image runs sim + render
        _run([local_blender, "-b", "--factory-startup", "--python",
              os.path.abspath(example_file), "--"] + tail)
    else:              # host: render inside the Docker image
        ensure_image()
        _run(["docker", "run", "--rm", "-v", MOUNT, IMAGE,
              "blender", "-b", "--factory-startup", "--python", "/work/" + rel, "--"] + tail)


def fresh_dir(work_rel):
    """Empty (and create) a directory under output/, given a '/work/output/...' path."""
    local = os.path.join(REPO, work_rel.replace("/work/", "").replace("/", os.sep))
    shutil.rmtree(local, ignore_errors=True)
    os.makedirs(local, exist_ok=True)


def make_gif(frames_pattern, gif_out, fps=20, width=640, period_s=None, n_frames=None):
    """frames_pattern/gif_out are '/work/...' paths; two-pass palette for clean
    colour. If `period_s`+`n_frames` are given, burn a mission clock (T+MM:SS of
    the simulated elapsed time, plus the orbital period) into the bottom-right."""
    pre = f"fps={fps},scale={width}:-1:flags=lanczos"
    if period_s and n_frames and n_frames > 1:
        sp = period_s / (n_frames - 1)                 # simulated seconds per frame
        # ffmpeg drawtext: ':' and ',' inside %{eif:...} must be backslash-escaped.
        total = f"{int(period_s) // 60:d}\\:{int(period_s) % 60:02d}"
        font = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
        s = f"{sp:.5f}"
        mm = "%{eif\\:trunc(n*" + s + "/60)\\:d\\:2}"
        ss = "%{eif\\:trunc(mod(n*" + s + "\\,60))\\:d\\:2}"
        text = "T+" + mm + "\\:" + ss + "  (orbit " + total + ")"
        fs = max(14, width // 38)
        pre += (f",drawtext=fontfile={font}:text='{text}'"
                f":x=w-tw-14:y=h-th-12:fontsize={fs}:fontcolor=white"
                ":box=1:boxcolor=black@0.5:boxborderw=8")
    vf = pre + ",split[s0][s1];[s0]palettegen=stats_mode=diff[p];[s1][p]paletteuse=dither=bayer"
    _run(["docker", "run", "--rm", "-v", MOUNT, IMAGE,
          "ffmpeg", "-y", "-framerate", str(fps), "-i", frames_pattern,
          "-vf", vf, "-loop", "0", gif_out])
