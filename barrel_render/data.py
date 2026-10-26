"""Data + astronomy helpers for the BARREL Blender renderer (NO bpy import).

Pure Python so the CSV parsing and the time/ephemeris math can be unit-tested
without Blender:

    python barrel_render/data.py [trajectory.csv]
"""

import csv
import math

TWO_PI = 2.0 * math.pi
ARCSEC = 1.0 / 206264.806247          # arcseconds -> radians

# WGS-84
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def _frac(x):
    return x - math.floor(x)


# --------------------------------------------------------------------------- #
# trajectory CSV
# --------------------------------------------------------------------------- #
def load_trajectory(path):
    """Load a BARREL CSV into {sat_id: {'t','jd','pos','quat'}}.

    Handles the single-satellite state.csv (no sat_id) and the multi-satellite
    trajectory.csv (with sat_id). Positions are ECI metres; quaternions are
    (x, y, z, w) body->inertial (identity if absent).
    """
    sats = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        has_id = "sat_id" in cols
        has_quat = "qw" in cols
        for row in reader:
            sid = int(float(row["sat_id"])) if has_id else 0
            s = sats.setdefault(sid, {"t": [], "jd": [], "pos": [], "quat": []})
            s["t"].append(float(row["t_s"]))
            s["jd"].append(float(row.get("jd_utc", 0.0)))
            s["pos"].append((float(row["x_m"]), float(row["y_m"]), float(row["z_m"])))
            if has_quat:
                s["quat"].append((float(row["qx"]), float(row["qy"]),
                                  float(row["qz"]), float(row["qw"])))
            else:
                s["quat"].append((0.0, 0.0, 0.0, 1.0))
    return sats


# --------------------------------------------------------------------------- #
# time / ephemerides
# --------------------------------------------------------------------------- #
def gmst(jd_ut1):
    """Greenwich Mean Sidereal Time (IAU-1982), radians in [0, 2*pi)."""
    t = (jd_ut1 - 2451545.0) / 36525.0
    sec = (67310.54841
           + (876600.0 * 3600.0 + 8640184.812866) * t
           + 0.093104 * t * t - 6.2e-6 * t * t * t)
    sec = math.fmod(sec, 86400.0)
    if sec < 0.0:
        sec += 86400.0
    return (sec * (TWO_PI / 86400.0)) % TWO_PI


def mean_obliquity(t):
    """Mean obliquity of the ecliptic (rad), t = Julian centuries TT since J2000."""
    return math.radians(23.43929111 - 0.0130042 * t - 1.64e-7 * t * t)


def sun_direction_eci(jd_tt):
    """Unit vector toward the Sun in ECI (low-precision, Montenbruck & Gill)."""
    t = (jd_tt - 2451545.0) / 36525.0
    m = TWO_PI * _frac(0.9931267 + 99.9973583 * t)
    lam = TWO_PI * _frac(0.7859444 + m / TWO_PI
                         + (6892.0 * math.sin(m) + 72.0 * math.sin(2.0 * m)) / 1.296e6)
    eps = mean_obliquity(t)
    x = math.cos(lam)
    y = math.sin(lam) * math.cos(eps)
    z = math.sin(lam) * math.sin(eps)
    n = math.sqrt(x * x + y * y + z * z)
    return (x / n, y / n, z / n)


def moon_position_eci(jd_tt):
    """Geocentric Moon position in ECI metres (Montenbruck & Gill low precision)."""
    t = (jd_tt - 2451545.0) / 36525.0
    l0 = _frac(0.606433 + 1336.851344 * t)
    l = TWO_PI * _frac(0.374897 + 1325.552410 * t)
    lp = TWO_PI * _frac(0.993133 + 99.997361 * t)
    d = TWO_PI * _frac(0.827361 + 1236.853086 * t)
    f = TWO_PI * _frac(0.259086 + 1342.227825 * t)

    dl = (22640.0 * math.sin(l) - 4586.0 * math.sin(l - 2 * d) + 2370.0 * math.sin(2 * d)
          + 769.0 * math.sin(2 * l) - 668.0 * math.sin(lp) - 412.0 * math.sin(2 * f)
          - 212.0 * math.sin(2 * l - 2 * d) - 206.0 * math.sin(l + lp - 2 * d)
          + 192.0 * math.sin(l + 2 * d) - 165.0 * math.sin(lp - 2 * d)
          + 148.0 * math.sin(l - lp) - 125.0 * math.sin(d) - 110.0 * math.sin(l + lp)
          - 55.0 * math.sin(2 * f - 2 * d))
    lam = TWO_PI * l0 + dl * ARCSEC

    s = f + (dl + 412.0 * math.sin(2 * f) + 541.0 * math.sin(lp)) * ARCSEC
    h = f - 2 * d
    n = (-526.0 * math.sin(h) + 44.0 * math.sin(l + h) - 31.0 * math.sin(-l + h)
         - 23.0 * math.sin(lp + h) + 11.0 * math.sin(-lp + h) - 25.0 * math.sin(-2 * l + f)
         + 21.0 * math.sin(-l + f))
    beta = (18520.0 * math.sin(s) + n) * ARCSEC

    r = (385000e3 - 20905e3 * math.cos(l) - 3699e3 * math.cos(2 * d - l)
         - 2956e3 * math.cos(2 * d) - 570e3 * math.cos(2 * l)
         + 246e3 * math.cos(2 * l - 2 * d) - 205e3 * math.cos(lp - 2 * d)
         - 171e3 * math.cos(l + 2 * d) - 152e3 * math.cos(l + lp - 2 * d))

    eps = mean_obliquity(t)
    cl, sl = math.cos(lam), math.sin(lam)
    cb, sb = math.cos(beta), math.sin(beta)
    ce, se = math.cos(eps), math.sin(eps)
    return (r * cb * cl,
            r * (ce * cb * sl - se * sb),
            r * (se * cb * sl + ce * sb))


# --------------------------------------------------------------------------- #
# geometry / colour
# --------------------------------------------------------------------------- #
def geodetic_to_ecef(lat_rad, lon_rad, alt_m=0.0):
    """WGS-84 geodetic (radians, metres) -> ECEF metres."""
    s, c = math.sin(lat_rad), math.cos(lat_rad)
    nph = WGS84_A / math.sqrt(1.0 - WGS84_E2 * s * s)
    return ((nph + alt_m) * c * math.cos(lon_rad),
            (nph + alt_m) * c * math.sin(lon_rad),
            (nph * (1.0 - WGS84_E2) + alt_m) * s)


# --------------------------------------------------------------------------- #
# attitude helpers
# --------------------------------------------------------------------------- #
def _finite_diff_velocity(pos, t, k):
    """Central finite-difference velocity of a position list at index k."""
    n = len(pos)
    if n < 2:
        return (1.0, 0.0, 0.0)
    if k <= 0:
        a, b, dt = pos[0], pos[1], (t[1] - t[0])
    elif k >= n - 1:
        a, b, dt = pos[n - 2], pos[n - 1], (t[n - 1] - t[n - 2])
    else:
        a, b, dt = pos[k - 1], pos[k + 1], (t[k + 1] - t[k - 1])
    dt = dt or 1.0
    return ((b[0] - a[0]) / dt, (b[1] - a[1]) / dt, (b[2] - a[2]) / dt)


def _dcm_to_quat(xb, yb, zb):
    """Body axes (as inertial columns) -> body->inertial quaternion (x,y,z,w)."""
    m00, m10, m20 = xb
    m01, m11, m21 = yb
    m02, m12, m22 = zb
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w, x, y, z = (m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w, x, y, z = (m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w, x, y, z = (m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    return (x / n, y / n, z / n, w / n)


def nadir_attitude(positions, times):
    """Nadir-pointing (LVLH) body->inertial quaternions (x,y,z,w), one per sample.

    Body +Z = zenith (+r_hat), so a camera's boresight (local -Z) looks nadir;
    body +X is the in-plane along-track direction. Useful as a visualization
    attitude when the trajectory carries no real attitude (orbit-only sims), so
    that body-mounted cameras image the ground.
    """
    quats = []
    for k, r in enumerate(positions):
        rn = math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]) or 1.0
        zb = (r[0] / rn, r[1] / rn, r[2] / rn)            # +Z = zenith
        v = _finite_diff_velocity(positions, times, k)
        vz = v[0] * zb[0] + v[1] * zb[1] + v[2] * zb[2]
        xb = (v[0] - vz * zb[0], v[1] - vz * zb[1], v[2] - vz * zb[2])
        xn = math.sqrt(xb[0] * xb[0] + xb[1] * xb[1] + xb[2] * xb[2]) or 1.0
        xb = (xb[0] / xn, xb[1] / xn, xb[2] / xn)         # +X = along-track
        yb = (zb[1] * xb[2] - zb[2] * xb[1],
              zb[2] * xb[0] - zb[0] * xb[2],
              zb[0] * xb[1] - zb[1] * xb[0])              # +Y = z x x
        quats.append(_dcm_to_quat(xb, yb, zb))
    return quats


# --------------------------------------------------------------------------- #
# flat-Earth projection (render-only Easter egg; dynamics untouched)
# --------------------------------------------------------------------------- #
def eci_to_ecef(p, jd):
    """Rotate an ECI position to ECEF: Rz(-GMST). p = (x, y, z) metres."""
    g = gmst(jd)
    c, s = math.cos(g), math.sin(g)
    x, y, z = p
    return (c * x + s * y, -s * x + c * y, z)


def flat_earth_point(p_eci, jd, height_scale=1.0):
    """Project an ECI position onto a flat (azimuthal-equidistant, North-Pole-
    centred) Earth. Returns (x, y, z) metres: (x, y) is the sub-point on the disk
    (longitude -> angle, so motion CIRCLES instead of jumping at the date line),
    z is the altitude above the disk (x height_scale). The disk edge (radius
    pi*R_earth) is the South Pole."""
    xe, ye, ze = eci_to_ecef(p_eci, jd)
    r = math.sqrt(xe * xe + ye * ye + ze * ze) or 1.0
    lat = math.asin(max(-1.0, min(1.0, ze / r)))
    lon = math.atan2(ye, xe)
    rho = WGS84_A * (math.pi / 2.0 - lat)        # 0 at N pole, pi*R at S pole
    alt = r - WGS84_A
    return (rho * math.cos(lon), rho * math.sin(lon), alt * height_scale)


# Viridis control colours (perceptually-uniform), for scatter colour mapping.
_VIRIDIS = [(0.267, 0.005, 0.329), (0.254, 0.265, 0.530), (0.128, 0.567, 0.551),
            (0.369, 0.789, 0.383), (0.993, 0.906, 0.144)]


def colormap(value01):
    """Map a value in [0,1] to an (r,g,b) viridis-like colour."""
    v = min(1.0, max(0.0, value01))
    x = v * (len(_VIRIDIS) - 1)
    i = int(math.floor(x))
    if i >= len(_VIRIDIS) - 1:
        return _VIRIDIS[-1]
    f = x - i
    a, b = _VIRIDIS[i], _VIRIDIS[i + 1]
    return (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2]))


def camera_intrinsics(res_x, res_y, fov_deg, fit="horizontal"):
    """Pinhole (fx, fy, cx, cy) for a Blender camera of FOV `fov_deg` at the given
    resolution. Blender's camera.angle fits the larger sensor dimension by default;
    pass fit="horizontal"/"vertical" to match."""
    base = res_x if fit == "horizontal" else res_y
    f = (base / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    return (f, f, res_x / 2.0, res_y / 2.0)


def disparity_to_depth(disparity, fx, baseline):
    """Stereo disparity (pixels) -> depth Z = fx*baseline/disparity (units of baseline)."""
    import numpy as np
    d = np.asarray(disparity, dtype=float)
    z = np.full(d.shape, float("inf"))
    nz = d > 1e-9
    z[nz] = fx * baseline / d[nz]
    return z


def backproject_depth(depth, fx, fy, cx, cy, cam_to_world,
                      stride=1, min_depth=1e-9, max_depth=None):
    """Back-project a depth map to an Nx3 point cloud in world coordinates (same units
    as cam_to_world's translation). Blender camera convention: +X right, +Y up, looking
    down local -Z; `depth` is positive distance along -Z. `cam_to_world` is the camera's
    4x4 world matrix. Used on a rendered/stereo-matched depth from EarthScene.render_stereo."""
    import numpy as np
    Z = np.asarray(depth, dtype=float)
    M = np.asarray(cam_to_world, dtype=float)
    h, w = Z.shape
    uu, vv = np.meshgrid(np.arange(0, w, stride), np.arange(0, h, stride))
    zz = Z[vv, uu]
    valid = np.isfinite(zz) & (zz > min_depth)
    if max_depth is not None:
        valid &= zz < max_depth
    uu, vv, zz = uu[valid], vv[valid], zz[valid]
    xc = (uu - cx) / fx * zz
    yc = -(vv - cy) / fy * zz
    zc = -zz
    pts_cam = np.stack([xc, yc, zc, np.ones_like(zz)], axis=0)
    return (M @ pts_cam)[:3].T


def _self_test(path=None):
    assert abs(math.degrees(gmst(2451545.0)) - 280.46061837) < 1e-4
    sx, sy, sz = sun_direction_eci(2451545.0)
    assert abs(math.sqrt(sx * sx + sy * sy + sz * sz) - 1.0) < 1e-9
    mx, my, mz = moon_position_eci(2451545.0)
    rmoon = math.sqrt(mx * mx + my * my + mz * mz)
    assert 3.4e8 < rmoon < 4.1e8, rmoon
    # Equator/prime-meridian sea level -> (a, 0, 0).
    ex, ey, ez = geodetic_to_ecef(0.0, 0.0, 0.0)
    assert abs(ex - WGS84_A) < 1e-6 and abs(ey) < 1e-6 and abs(ez) < 1e-6
    c = colormap(0.5)
    assert all(0.0 <= ch <= 1.0 for ch in c)
    # Nadir attitude: body -Z must point at Earth centre (= -r_hat).
    q = nadir_attitude([(7.0e6, 0.0, 0.0), (7.0e6, 7.5e3, 0.0)], [0.0, 10.0])[0]
    x, y, z, w = q
    assert abs(x * x + y * y + z * z + w * w - 1.0) < 1e-9
    # body -Z rotated into inertial = R @ (0,0,-1); rebuild R's 3rd column.
    zcol = (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y))
    assert zcol[0] > 0.99, zcol            # +Z_body = +x_hat (zenith) for r=(+x)
    print(f"OK  gmst(J2000)={math.degrees(gmst(2451545.0)):.5f} deg, "
          f"moon r={rmoon/1e3:.0f} km, mid-colormap={tuple(round(x,2) for x in c)}")
    if path:
        sats = load_trajectory(path)
        any_id = next(iter(sats))
        print(f"OK  loaded {len(sats)} satellite(s), {len(sats[any_id]['t'])} samples from {path}")


if __name__ == "__main__":
    import sys
    _self_test(sys.argv[1] if len(sys.argv) > 1 else None)
