"""Local texture lookup and Blender execution for BARREL missions."""

import os
import shutil
import subprocess
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[1])

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr")


def asset(rel):
    """Absolute path for a repo-relative asset if it exists, else ''."""
    return os.path.join(REPO, rel) if os.path.exists(os.path.join(REPO, rel)) else ""


def _first_image(absdir):
    """Absolute path of the first image file in a directory, else ''."""
    if os.path.isdir(absdir):
        for fn in sorted(os.listdir(absdir)):
            if fn.lower().endswith(_IMG_EXTS):
                rel = os.path.relpath(os.path.join(absdir, fn), REPO).replace("\\", "/")
                return os.path.join(REPO, rel)
    return ""


def earth_maps(res="8k", folder="Texture_Files"):
    """Resolve Earth maps in Texture_Files/Earth-<res>/ by filename keyword.
    Returns absolute paths for color/bump/normal/night/specular
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
                out[key] = os.path.join(REPO, rel, fn)
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



def render(script, args):
    """Run a BARREL render script with the Blender installed in this environment."""
    blender = shutil.which("blender")
    if blender is None:
        raise RuntimeError("Blender must be installed to render mission frames.")
    subprocess.run([blender, "-b", "--factory-startup", "--python",
                    os.path.abspath(script), "--", *map(str, args)], check=True)
