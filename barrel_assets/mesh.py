"""Minimal binary/ASCII STL reader and surface point sampler (numpy only).

Used by the template-pose figures to load NASA-3D-Resources satellite meshes as templates.
"""
import struct
import numpy as np


def read_stl(path):
    """Return triangle vertices (n, 3, 3) from a binary or ASCII STL file."""
    with open(path, "rb") as f:
        head = f.read(80)
        rest = f.read()
    # ASCII STL begins with "solid" and contains no binary triangle block of the right size
    if head[:5].lower() == b"solid" and b"facet" in rest[:2048].lower():
        txt = (head + rest).decode("ascii", "ignore")
        verts = []
        for line in txt.splitlines():
            s = line.strip().split()
            if len(s) == 4 and s[0] == "vertex":
                verts.append([float(s[1]), float(s[2]), float(s[3])])
        v = np.asarray(verts, dtype=float)
        return v[: len(v) // 3 * 3].reshape(-1, 3, 3)
    n = struct.unpack("<I", rest[:4])[0]
    data = np.frombuffer(rest[4:4 + 50 * n], dtype=np.uint8).reshape(n, 50)
    return data[:, 12:48].copy().view("<f4").reshape(n, 3, 3).astype(float)


def centered_mesh(path):
    """Read a metre-scale mesh and place its surface-area centroid at the origin."""
    tri = read_stl(path)
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0],
                                   tri[:, 2] - tri[:, 0]), axis=1)
    origin = np.average(tri.mean(axis=1), axis=0, weights=area)
    return tri - origin


def sample_surface(tri, n_pts, seed=0, return_normals=False):
    """Sample (n_pts, 3) surface points and optionally their mesh-face normals."""
    rng = np.random.default_rng(seed)
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    area = np.where(np.isfinite(area), area, 0.0)
    if area.sum() <= 0:
        if return_normals:
            raise ValueError("Surface normals require nondegenerate mesh faces")
        pts = tri.reshape(-1, 3)
        idx = rng.choice(len(pts), size=n_pts)
        P = pts[idx]
    else:
        fidx = rng.choice(len(tri), size=n_pts, p=area / area.sum())
        u = rng.random(n_pts); w = rng.random(n_pts)
        over = u + w > 1.0
        u[over] = 1 - u[over]; w[over] = 1 - w[over]
        P = v0[fidx] + u[:, None] * (v1[fidx] - v0[fidx]) + w[:, None] * (v2[fidx] - v0[fidx])
        if return_normals:
            normals = np.cross(v1[fidx] - v0[fidx], v2[fidx] - v0[fidx])
            normals /= np.linalg.norm(normals, axis=1, keepdims=True)
            return P, normals
    return P
