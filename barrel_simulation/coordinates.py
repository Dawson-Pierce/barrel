"""Coordinate, attitude, and analytic astronomy functions used by the simulator."""

import math

import numpy as np

TWO_PI = 2.0 * math.pi

ARCSEC = 1.0 / 206264.806247

WGS84_A = 6378137.0

WGS84_F = 1.0 / 298.257223563

WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

def _frac(x):
    return x - math.floor(x)

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

def geodetic_to_ecef(lat_rad, lon_rad, alt_m=0.0):
    """WGS-84 geodetic (radians, metres) -> ECEF metres."""
    s, c = math.sin(lat_rad), math.cos(lat_rad)
    nph = WGS84_A / math.sqrt(1.0 - WGS84_E2 * s * s)
    return ((nph + alt_m) * c * math.cos(lon_rad),
            (nph + alt_m) * c * math.sin(lon_rad),
            (nph * (1.0 - WGS84_E2) + alt_m) * s)

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

def eci_to_ecef(p, jd):
    """Rotate an ECI position to ECEF: Rz(-GMST). p = (x, y, z) metres."""
    g = gmst(jd)
    c, s = math.cos(g), math.sin(g)
    x, y, z = p
    return (c * x + s * y, -s * x + c * y, z)

def quat_to_dcm(q):
    x, y, z, w = np.asarray(q, dtype=float)
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])
