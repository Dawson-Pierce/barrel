"""The Mission class: own a constellation, propagate it with the Basilisk astrodynamics
framework at full barrelsim fidelity, log CSVs, and render -- all into a run-scoped folder.

Basilisk owns the dynamics (the old barrelsim C++ core is retired) but the modelled physics
match what barrelsim did:

  * geopotential -- spherical harmonics (GGM03S) to degree/order `gravity_degree`,
  * drag         -- exponential atmosphere (Vallado layer at the orbit altitude) + cannonball,
  * third body   -- Sun and Moon point-mass perturbations from analytic ephemerides,
  * attitude     -- three reaction wheels per satellite slaved to an MRP feedback controller
                    tracking a guidance reference (nadir / sun / target / inertial hold).

The renderer (barrel_render) is untouched: the Mission writes the same trajectory.csv contract
(sat_id, t_s, jd_utc, x_m, y_m, z_m, qx..qw, ECI metres, body->inertial quaternion). The logged
quaternion is the satellite's *controlled* attitude, in the renderer's body convention
(body +Z = zenith, so a camera's -Z boresight images whatever the pointing mode targets).

Propagation is incremental: run() advances every satellite by `duration` from its current
state, so commands (pointing, impulsive delta-v) can be issued between run() calls. The dynamics
integrate (and the controllers run) at a fixed fine step `control_dt`; run()'s `dt` is only the
logging cadence.
"""

import csv
import math
import os
from datetime import datetime

import numpy as np
from Basilisk.architecture import messaging, sysModel
from Basilisk.fswAlgorithms import attTrackingError, mrpFeedback, rwMotorTorque
from Basilisk.simulation import (dragDynamicEffector, exponentialAtmosphere, gravityEffector,
                                  reactionWheelStateEffector, simpleNav, spacecraft)
from Basilisk.utilities import RigidBodyKinematics as rbk
from Basilisk.utilities import (SimulationBaseClass, macros, orbitalMotion, simIncludeGravBody,
                                 simIncludeRW)

from barrel_render import data as bdata          # bpy-free astronomy/CSV helpers

# Earth constants (match Basilisk's gravBodyFactory.createEarth()).
MU_EARTH = 0.39860043550702266e15               # m^3/s^2
R_EARTH = 6378136.6                              # m (equatorial)
DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# The Montenbruck-Gill / unnormalized recursion in any geopotential evaluator loses accuracy
# at very high degree; degree 40 captures the dynamically significant LEO field (terms past
# ~deg 30 attenuate as (R/r)^n), so requests above this are clamped.
_MAX_SH_DEGREE = 40

# Vallado, Fundamentals of Astrodynamics, Table 8-4: piecewise-exponential atmosphere
# (base altitude km, nominal density kg/m^3, scale height km). Basilisk's exponential
# atmosphere is single-layer, so we anchor it to the layer covering the orbit altitude.
_VALLADO_LAYERS = [
    (0, 1.225, 7.249), (25, 3.899e-2, 6.349), (30, 1.774e-2, 6.682), (40, 3.972e-3, 7.554),
    (50, 1.057e-3, 8.382), (60, 3.206e-4, 7.714), (70, 8.770e-5, 6.549), (80, 1.905e-5, 5.799),
    (90, 3.396e-6, 5.382), (100, 5.297e-7, 5.877), (110, 9.661e-8, 7.263), (120, 2.438e-8, 9.473),
    (130, 8.484e-9, 12.636), (140, 3.845e-9, 16.149), (150, 2.070e-9, 22.523),
    (180, 5.464e-10, 29.740), (200, 2.789e-10, 37.105), (250, 7.248e-11, 45.546),
    (300, 2.418e-11, 53.628), (350, 9.518e-12, 53.298), (400, 3.725e-12, 58.515),
    (450, 1.585e-12, 60.828), (500, 6.967e-13, 63.822), (600, 1.454e-13, 71.835),
    (700, 3.614e-14, 88.667), (800, 1.170e-14, 124.64), (900, 5.245e-15, 181.05),
    (1000, 3.019e-15, 268.00),
]


def _jd_from_calendar(y, mo, d, h=0, mi=0, s=0.0):
    """UTC Gregorian calendar date to Julian Date."""
    year = int(y)
    month = int(mo)
    if month <= 2:
        year -= 1
        month += 12
    century = year // 100
    correction = 2 - century + century // 4
    day_start = (
        math.floor(365.25 * (year + 4716))
        + math.floor(30.6001 * (month + 1))
        + int(d)
        + correction
        - 1524.5
    )
    return day_start + (float(h) + float(mi) / 60.0 + float(s) / 3600.0) / 24.0


def _sun_position_eci(jd_tt):
    """Geocentric Sun position in ECI metres (Montenbruck & Gill low precision): the renderer
    only keeps the direction, so reattach the analytic distance for third-body gravity."""
    t = (jd_tt - 2451545.0) / 36525.0
    m = bdata.TWO_PI * bdata._frac(0.9931267 + 99.9973583 * t)
    r = 149.619e9 - 2.499e9 * math.cos(m) - 0.021e9 * math.cos(2.0 * m)
    ux, uy, uz = bdata.sun_direction_eci(jd_tt)
    return (r * ux, r * uy, r * uz)


def _load_sh_coeffs(path, max_deg):
    """Parse an ICGEM-style coefficient file (gm/radius header + 'n m Cbar Sbar' rows, fully
    normalized) into (gm, radius, cBar, sBar, deg) for a Basilisk SphericalHarmonicsGravityModel."""
    gm = radius = None
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("gm"):
                gm = float(line.split("=")[1])
            elif line.startswith("radius"):
                radius = float(line.split("=")[1])
            else:
                p = line.split()
                if len(p) >= 4 and int(p[0]) <= max_deg:
                    rows.append((int(p[0]), int(p[1]), float(p[2]), float(p[3])))
    deg = max(n for n, *_ in rows)
    cBar = [[0.0] * (deg + 1) for _ in range(deg + 1)]
    sBar = [[0.0] * (deg + 1) for _ in range(deg + 1)]
    for n, m, c, s in rows:
        cBar[n][m] = c
        sBar[n][m] = s
    return gm, radius, cBar, sBar, deg


class _TranslationReference(sysModel.SysModel):
    """Publish precomputed orbital states before each attitude dynamics step."""

    def __init__(self, times, positions, velocities):
        super().__init__()
        self.times = np.asarray(times)
        self.positions = np.asarray(positions)
        self.velocities = np.asarray(velocities)
        self.messages = [messaging.TransRefMsg() for _ in range(self.positions.shape[1])]
        self.UpdateState(0)

    def UpdateState(self, current_time):
        index = int(np.searchsorted(self.times, current_time * 1e-9 - 1e-8, side="left"))
        if index >= len(self.times) or abs(self.times[index] - current_time * 1e-9) > 1e-8:
            raise ValueError("prescribed translation must cover every attitude dynamics step")
        for sat, message in enumerate(self.messages):
            payload = messaging.TransRefMsgPayload()
            payload.r_RN_N = self.positions[index, sat].tolist()
            payload.v_RN_N = self.velocities[index, sat].tolist()
            message.write(payload, current_time, self.moduleID)


class _Guidance(sysModel.SysModel):
    """A Basilisk Python module that publishes the reference attitude each control step.

    Reads the satellite's navigated translation state and writes an AttRefMsg for the active
    pointing mode, in the renderer's body convention: the camera boresight (body -Z) points at
    the look direction L (nadir / sun / a ground target), body +X along-track. Inertial hold
    freezes the body frame in inertial space."""

    def __init__(self, jd0):
        super().__init__()
        self.jd0 = jd0
        self.transInMsg = messaging.NavTransMsgReader()
        self.attRefOutMsg = messaging.AttRefMsg()
        self.mode = "nadir"
        self.target_lla = None          # (lat_rad, lon_rad, alt_m) for "target"
        self.target_eci = None          # explicit inertial target (overrides target_lla)

    def reference(self, r, v, t_sec):
        """(sigma_RN, omega_RN_N) for the active mode at sim time t_sec (s)."""
        if self.mode == "inertial":
            return np.zeros(3), np.zeros(3)
        rhat = r / np.linalg.norm(r)
        jd = self.jd0 + t_sec / 86400.0
        if self.mode == "sun":
            look = np.array(bdata.sun_direction_eci(jd))
        elif self.mode == "target":
            tgt = self._target_eci(t_sec)
            look = (tgt - r) / np.linalg.norm(tgt - r) if tgt is not None else -rhat
        else:                                              # nadir
            look = -rhat
        zb = -look / np.linalg.norm(look)                  # body +Z opposite the boresight
        xb = v - np.dot(v, zb) * zb                        # body +X along-track
        n = np.linalg.norm(xb)
        xb = xb / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
        yb = np.cross(zb, xb)
        RN = np.row_stack([xb, yb, zb])                    # inertial -> reference (rows = body axes)
        omega = np.cross(r, v) / np.dot(r, r) if self.mode == "nadir" else np.zeros(3)
        return rbk.C2MRP(RN), omega

    def _target_eci(self, t_sec):
        if self.target_eci is not None:
            return np.asarray(self.target_eci, float)
        if self.target_lla is None:
            return None
        ecef = np.array(bdata.geodetic_to_ecef(*self.target_lla))
        g = bdata.gmst(self.jd0 + t_sec / 86400.0)         # ECEF -> ECI = Rz(+gmst)
        c, s = math.cos(g), math.sin(g)
        return np.array([c * ecef[0] - s * ecef[1], s * ecef[0] + c * ecef[1], ecef[2]])

    def UpdateState(self, t):
        st = self.transInMsg()
        sigma_RN, omega = self.reference(np.array(st.r_BN_N), np.array(st.v_BN_N),
                                         t * macros.NANO2SEC)
        pl = messaging.AttRefMsgPayload()
        pl.sigma_RN = list(sigma_RN)
        pl.omega_RN_N = list(omega)
        pl.domega_RN_N = [0.0, 0.0, 0.0]
        self.attRefOutMsg.write(pl, t, self.moduleID)


class Mission:
    """A constellation + its commands + Basilisk propagation + rendering, with its own
    timestamped output folder."""

    def __init__(self, name=None, output_root="output", epoch=(2026, 1, 1, 0, 0, 0),
                 earth_res="8k", forces=("j2", "drag", "sun", "moon"), gravity_degree=32,
                 gravity_file=None, control_dt=1.0):
        self.earth_res = earth_res
        self.epoch = tuple(epoch)
        self.jd0 = _jd_from_calendar(*epoch)
        self._forces = set(forces)
        self._gravity_degree = int(gravity_degree)
        self._gravity_file = gravity_file or os.path.join(_DATA_DIR, "gravity", "ggm03s.txt")
        self._control_dt = float(control_dt)

        stamp = name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.run_dir = os.path.abspath(os.path.join(output_root, stamp))
        os.makedirs(self.run_dir, exist_ok=True)

        self._specs = []          # per-satellite request dicts (pre-build)
        self._sc = []             # Basilisk Spacecraft objects (post-build)
        self._recs = []           # state recorders
        self._guid = []           # per-sat guidance modules
        self._third = []          # (gravBody, message, ephem_fn) for Sun/Moon
        self._built = False
        self._log_dt = None
        self._elapsed = 0.0
        self._translation = None
        self.frames_logged = 0
        import Basilisk
        print(f"[mission] run dir: {self.run_dir}  (Basilisk {Basilisk.__version__})")

    # ---- setup -------------------------------------------------------------
    def prescribe_translation(self, times, positions, velocities):
        """Use supplied ECI states while integrating reaction wheels and attitude.

        Supply one state per satellite at every ``control_dt`` epoch, starting at
        zero. Orbital forces belong to the propagation that produced these states.
        """
        if self._built:
            raise RuntimeError("prescribe translation before starting the mission")
        times = np.asarray(times, dtype=float)
        positions, velocities = np.asarray(positions), np.asarray(velocities)
        shape = (len(times), len(self._specs), 3)
        if positions.shape != shape or velocities.shape != shape or len(times) == 0:
            raise ValueError("translation arrays must have shape (times, satellites, 3)")
        if not np.allclose(times, np.arange(len(times)) * self._control_dt, rtol=0, atol=1e-8):
            raise ValueError("translation times must match the attitude dynamics cadence")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
            raise ValueError("translation states must be finite")
        self._translation = _TranslationReference(times, positions, velocities)

    def add_satellite(self, elements=None, state=None, name=None, pointing="nadir",
                      cameras=(("overview",),), mass=100.0, drag_area=1.0, drag_cd=2.2,
                      inertia=(10.0, 10.0, 10.0), wheels=True, **_ignored):
        """Add a satellite from classical elements (a,e,i,raan,argp,nu; SI/rad) or a raw
        (r, v) ECI state. `cameras` is a list of specs, e.g.
        [("overview",), ("sensor", fov_deg, off_nadir_deg)]. `inertia` is the body principal
        moments (kg m^2); `wheels` enables the three-wheel attitude controller."""
        name = name or f"sat{len(self._specs)}"
        self._specs.append(dict(name=name, cameras=[tuple(c) for c in cameras],
                                pointing=pointing, elements=elements, state=state,
                                mass=float(mass), drag_area=float(drag_area), drag_cd=float(drag_cd),
                                inertia=tuple(float(x) for x in inertia), wheels=bool(wheels)))
        return len(self._specs) - 1

    def _initial_rv(self, spec):
        if spec["state"] is not None:
            r, v = spec["state"]
            return np.array(list(map(float, r))), np.array(list(map(float, v)))
        oe = orbitalMotion.ClassicElements()
        a, e, i, raan, argp, nu = spec["elements"]
        oe.a, oe.e, oe.i, oe.Omega, oe.omega, oe.f = a, e, i, raan, argp, nu
        r, v = orbitalMotion.elem2rv(MU_EARTH, oe)
        return np.array(r), np.array(v)

    # ---- world (gravity, atmosphere, third bodies) -------------------------
    def _build_gravity(self):
        """Earth gravity body: spherical harmonics if requested and the coeff file exists,
        else point mass."""
        gf = simIncludeGravBody.gravBodyFactory()
        earth = gf.createEarth()
        earth.isCentralBody = True
        use_sh = ("j2" in self._forces or "sh" in self._forces) and os.path.exists(self._gravity_file)
        if use_sh:
            deg = min(self._gravity_degree, _MAX_SH_DEGREE)
            gm, radius, cBar, sBar, deg = _load_sh_coeffs(self._gravity_file, deg)
            shm = gravityEffector.SphericalHarmonicsGravityModel()
            shm.muBody = gm
            shm.radEquator = radius
            shm.maxDeg = deg
            shm.cBar = cBar
            shm.sBar = sBar
            earth.gravityModel = shm
            print(f"[mission] gravity: {os.path.basename(self._gravity_file)} "
                  f"spherical harmonics deg/order {deg}")
        else:
            print("[mission] gravity: point mass")
        return gf

    def _build_third_bodies(self, gf):
        """Sun/Moon gravity bodies fed by stand-alone ephemeris messages (no SPICE kernels);
        the messages are refreshed each propagation segment from the analytic ephemerides."""
        wanted = []
        if "sun" in self._forces:
            wanted.append(("sun", gf.createSun(), _sun_position_eci))
        if "moon" in self._forces:
            wanted.append(("moon", gf.createMoon(), bdata.moon_position_eci))
        for label, body, ephem in wanted:
            msg = messaging.SpicePlanetStateMsg()
            body.planetBodyInMsg.subscribeTo(msg)
            self._third.append((body, msg, ephem))
        if wanted:
            print(f"[mission] third body: {', '.join(w[0] for w in wanted)}")

    def _refresh_third_bodies(self):
        """Write the current Sun/Moon positions into their ephemeris messages."""
        jd = self.jd0 + self._elapsed / 86400.0
        for body, msg, ephem in self._third:
            pl = messaging.SpicePlanetStateMsgPayload()
            pl.PlanetName = body.planetName
            pl.PositionVector = list(map(float, ephem(jd)))
            pl.VelocityVector = [0.0, 0.0, 0.0]
            pl.J20002Pfix = np.identity(3).tolist()
            msg.write(pl)

    def _build_atmosphere(self):
        """Exponential atmosphere anchored to the Vallado layer at the mean orbit altitude, so
        the single-layer Basilisk model reproduces barrelsim's piecewise table near the orbit."""
        alts = [np.linalg.norm(self._initial_rv(s)[0]) - R_EARTH for s in self._specs]
        h_km = max(0.0, float(np.mean(alts)) / 1000.0)
        h0, rho0, H = _VALLADO_LAYERS[0]
        for b, r0, sh in _VALLADO_LAYERS:
            if h_km >= b:
                h0, rho0, H = b, r0, sh
            else:
                break
        atmo = exponentialAtmosphere.ExponentialAtmosphere()
        atmo.ModelTag = "atmo"
        atmo.planetRadius = R_EARTH
        atmo.scaleHeight = H * 1000.0
        atmo.baseDensity = rho0 * math.exp((h0 * 1000.0) / (H * 1000.0))   # rho(h0)=rho0
        print(f"[mission] drag: exponential atmosphere anchored at {h0} km "
              f"(rho={rho0:.3e}, H={H} km)")
        return atmo

    # ---- attitude (reaction wheels + MRP feedback) -------------------------
    def _build_attitude(self, sc, spec):
        """Wire one satellite's three reaction wheels and the FSW chain
        (simpleNav -> guidance -> attTrackingError -> mrpFeedback -> rwMotorTorque -> wheels)."""
        rwFactory = simIncludeRW.rwFactory()
        for ax in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
            w = rwFactory.create("Honeywell_HR16", ax, maxMomentum=50.0)
            w.Js = 0.01           # spin inertia (kg m^2), barrelsim wheel
            w.u_max = 0.05        # max motor torque (N m)
            w.Omega_max = 100.0   # rad/s, generous so nadir tracking never momentum-saturates
        rwEff = reactionWheelStateEffector.ReactionWheelStateEffector()
        rwFactory.addToSpacecraft(sc.ModelTag + "_rw", rwEff, sc)
        rwParamMsg = rwFactory.getConfigMessage()

        nav = simpleNav.SimpleNav()
        nav.ModelTag = sc.ModelTag + "_nav"
        nav.scStateInMsg.subscribeTo(sc.scStateOutMsg)

        guid = _Guidance(self.jd0)
        guid.ModelTag = sc.ModelTag + "_guid"
        guid.mode = spec["pointing"]
        guid.transInMsg.subscribeTo(nav.transOutMsg)

        attErr = attTrackingError.attTrackingError()
        attErr.ModelTag = sc.ModelTag + "_attErr"
        attErr.attNavInMsg.subscribeTo(nav.attOutMsg)
        attErr.attRefInMsg.subscribeTo(guid.attRefOutMsg)

        I = spec["inertia"]
        vehCfg = messaging.VehicleConfigMsgPayload()
        vehCfg.ISCPntB_B = [I[0], 0., 0., 0., I[1], 0., 0., 0., I[2]]
        vehCfgMsg = messaging.VehicleConfigMsg().write(vehCfg)

        fb = mrpFeedback.mrpFeedback()
        fb.ModelTag = sc.ModelTag + "_mrp"
        fb.K = 0.5
        fb.P = 5.0
        fb.Ki = -1.0              # negative -> integral term disabled (pure PD, like barrelsim)
        fb.integralLimit = 0.0
        fb.guidInMsg.subscribeTo(attErr.attGuidOutMsg)
        fb.vehConfigInMsg.subscribeTo(vehCfgMsg)
        fb.rwParamsInMsg.subscribeTo(rwParamMsg)
        fb.rwSpeedsInMsg.subscribeTo(rwEff.rwSpeedOutMsg)

        rwMot = rwMotorTorque.rwMotorTorque()
        rwMot.ModelTag = sc.ModelTag + "_rwMot"
        rwMot.controlAxes_B = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        rwMot.vehControlInMsg.subscribeTo(fb.cmdTorqueOutMsg)
        rwMot.rwParamsInMsg.subscribeTo(rwParamMsg)
        rwEff.rwMotorCmdInMsg.subscribeTo(rwMot.rwMotorTorqueOutMsg)

        # Every object here is referenced by the running C++ task, so the dict (stored on the
        # Mission in _build) must outlive the sim: a GC'd RW-config message or FSW module leaves
        # the task reading freed memory -> singular matrices / garbage torques.
        return dict(rwEff=rwEff, rwFactory=rwFactory, rwParamMsg=rwParamMsg, nav=nav, guid=guid,
                    attErr=attErr, vehCfgMsg=vehCfgMsg, fb=fb, rwMot=rwMot,
                    fsw=[nav, guid, attErr, fb, rwMot])

    def _initial_attitude(self, spec, r, v):
        """sigma_BNInit so the satellite starts already on its pointing reference (no slew)."""
        g = _Guidance(self.jd0)
        g.mode = spec["pointing"]
        if spec["pointing"] == "target" and spec.get("target_lla"):
            g.target_lla = spec["target_lla"]
        sigma_RN, _ = g.reference(np.asarray(r, float), np.asarray(v, float), 0.0)
        return sigma_RN

    def _build(self, dt):
        self._log_dt = float(dt)
        self._scSim = SimulationBaseClass.SimBaseClass()
        proc = self._scSim.CreateNewProcess("dyn")
        proc.addTask(self._scSim.CreateNewTask("dynTask", macros.sec2nano(self._control_dt)))

        gf = self._build_gravity() if self._translation is None else None
        if gf is not None:
            self._build_third_bodies(gf)
        atmo = self._build_atmosphere() if "drag" in self._forces and gf is not None else None
        if self._translation is not None:
            self._scSim.AddModelToTask("dynTask", self._translation, 100)
        # Hold the gravity factory + atmosphere on self: the spacecraft tasks reference their
        # C++ objects, so they must not be garbage collected after _build returns.
        self._gf = gf
        self._atmo = atmo

        atts = []
        for sat_index, spec in enumerate(self._specs):
            sc = spacecraft.Spacecraft()
            sc.ModelTag = spec["name"]
            sc.hub.mHub = spec["mass"]
            I = spec["inertia"]
            sc.hub.IHubPntBc_B = [[I[0], 0., 0.], [0., I[1], 0.], [0., 0., I[2]]]
            r, v = self._initial_rv(spec)
            sc.hub.r_CN_NInit = r.tolist()
            sc.hub.v_CN_NInit = v.tolist()
            if spec["wheels"]:
                sc.hub.sigma_BNInit = [[x] for x in self._initial_attitude(spec, r, v)]
            if gf is not None:
                gf.addBodiesTo(sc)
            else:
                sc.transRefInMsg.subscribeTo(self._translation.messages[sat_index])
            self._scSim.AddModelToTask("dynTask", sc)
            self._sc.append(sc)

            if atmo is not None:
                atmo.addSpacecraftToModel(sc.scStateOutMsg)
                drag = dragDynamicEffector.DragDynamicEffector()
                drag.ModelTag = spec["name"] + "_drag"
                drag.coreParams.projectedArea = spec["drag_area"]
                drag.coreParams.dragCoeff = spec["drag_cd"]
                drag.atmoDensInMsg.subscribeTo(atmo.envOutMsgs[-1])
                # Spacecraft.Reset evaluates forces before the first scheduled
                # drag update. Prime Basilisk's otherwise uninitialized density
                # cache from the message's zero payload before that evaluation.
                drag.ReadInputs()
                sc.addDynamicEffector(drag)
                spec["_drag"] = drag

            atts.append(self._build_attitude(sc, spec) if spec["wheels"] else None)

        if atmo is not None:
            self._scSim.AddModelToTask("dynTask", atmo)
            for spec in self._specs:
                if "_drag" in spec:
                    self._scSim.AddModelToTask("dynTask", spec["_drag"])

        self._atts = atts                                  # keep FSW + wheel objects alive
        for att in atts:                                   # FSW + wheels after the dynamics
            self._guid.append(att["guid"] if att is not None else None)   # index-aligned with sats
            if att is None:
                continue
            self._scSim.AddModelToTask("dynTask", att["rwEff"])
            for mod in att["fsw"]:
                self._scSim.AddModelToTask("dynTask", mod)

        for sc in self._sc:
            rec = sc.scStateOutMsg.recorder(macros.sec2nano(self._log_dt))
            self._scSim.AddModelToTask("dynTask", rec)
            self._recs.append(rec)

        self._refresh_third_bodies()
        self._scSim.InitializeSimulation()
        self._built = True

    # ---- commands ----------------------------------------------------------
    def _resolve(self, sat):
        if isinstance(sat, str):
            for i, s in enumerate(self._specs):
                if s["name"] == sat:
                    return i
            raise KeyError(f"no satellite named {sat!r}")
        return int(sat)

    def command(self, sat, pointing=None, target_latlon=None, target_eci=None, delta_v_eci=None):
        """Change a satellite's pointing mode, aim it at a ground target (lat,lon[,alt] deg/m)
        or an inertial point, and/or apply an impulsive delta-v (m/s, ECI) to the live state."""
        i = self._resolve(sat)
        spec = self._specs[i]
        guid = self._guid[i] if (self._built and i < len(self._guid) and spec["wheels"]) else None
        if pointing:
            spec["pointing"] = pointing
            if guid is not None:
                guid.mode = pointing
        if target_latlon is not None:
            lat, lon = target_latlon[0], target_latlon[1]
            alt = target_latlon[2] if len(target_latlon) > 2 else 0.0
            spec["pointing"] = "target"
            spec["target_lla"] = (lat * DEG2RAD, lon * DEG2RAD, alt)
            if guid is not None:
                guid.mode = "target"
                guid.target_lla = spec["target_lla"]
        if target_eci is not None:
            spec["pointing"] = "target"
            if guid is not None:
                guid.mode = "target"
                guid.target_eci = list(target_eci)
        if delta_v_eci is not None:
            if not self._built:
                raise RuntimeError("delta-v requires the sim to be running (call run() first)")
            vState = self._sc[i].dynManager.getStateObject(self._sc[i].hub.nameOfHubVelocity)
            v = np.array(vState.getState()).reshape(-1) + np.asarray(delta_v_eci, dtype=float)
            vState.setState(v.reshape(3, 1).tolist())

    # ---- propagation -------------------------------------------------------
    def run(self, duration, dt=30.0, integ_step=None):
        """Advance every satellite by `duration` from its current state, logging every `dt`
        seconds (the dynamics + controllers always integrate at the fine `control_dt` step).
        Repeated calls continue the propagation."""
        if not self._built:
            self._build(dt)
        self._refresh_third_bodies()
        self._elapsed += float(duration)
        self._scSim.ConfigureStopTime(macros.sec2nano(self._elapsed))
        self._scSim.ExecuteSimulation()

    def step(self, dt, integ_step=None):
        self.run(dt, dt)

    def state(self, sat):
        """Current live truth state for the event-driven BARREL scene."""
        if not self._built:
            raise RuntimeError("mission has not been advanced; call step() first")
        i = self._resolve(sat)
        payload = self._sc[i].scStateOutMsg.read()
        sigma = np.asarray(payload.sigma_BN, dtype=float)
        # MRP2C is inertial->body; the scene and sensor mounting contract uses body->ECI.
        rotation = rbk.MRP2C(sigma).T
        return {
            "position": np.asarray(payload.r_BN_N, dtype=float),
            "velocity": np.asarray(payload.v_BN_N, dtype=float),
            "rotation": rotation,
            "omega_body": np.asarray(payload.omega_BN_B, dtype=float),
            "jd": self.jd0 + self._elapsed / 86400.0,
            "time_s": self._elapsed,
        }

    # ---- logging -----------------------------------------------------------
    def _quat_bodytoinertial(self, sigma_BN):
        """Controlled MRP attitude -> body->inertial quaternion (x,y,z,w) in the renderer's
        convention (reuse the renderer's DCM->quat so cameras stay consistent)."""
        BN = rbk.MRP2C(np.asarray(sigma_BN, float))        # inertial -> body; rows = body axes in N
        return bdata._dcm_to_quat(tuple(BN[0]), tuple(BN[1]), tuple(BN[2]))

    def _write(self):
        if not self._built or self.frames_logged:
            return
        tf = open(os.path.join(self.run_dir, "trajectory.csv"), "w", newline="")
        tw = csv.writer(tf)
        tw.writerow(["sat_id", "t_s", "jd_utc", "x_m", "y_m", "z_m", "qx", "qy", "qz", "qw"])
        af = open(os.path.join(self.run_dir, "analysis.csv"), "w", newline="")
        aw = csv.writer(af)
        aw.writerow(["sat_id", "name", "t_s", "jd_utc", "x", "y", "z", "vx", "vy", "vz",
                     "qx", "qy", "qz", "qw", "wx", "wy", "wz",
                     "a", "e", "inc_deg", "raan_deg", "argp_deg", "nu_deg"])
        for i, (spec, rec) in enumerate(zip(self._specs, self._recs)):
            t = np.asarray(rec.times()) * macros.NANO2SEC
            r = np.asarray(rec.r_BN_N)
            v = np.asarray(rec.v_BN_N)
            sig = np.asarray(rec.sigma_BN)
            wB = np.asarray(rec.omega_BN_B)
            for k in range(len(t)):
                if not (np.all(np.isfinite(r[k])) and np.all(np.isfinite(v[k]))):
                    continue
                jd = self.jd0 + t[k] / 86400.0
                q = self._quat_bodytoinertial(sig[k]) if spec["wheels"] else (0.0, 0.0, 0.0, 1.0)
                tw.writerow([i, f"{t[k]:.3f}", f"{jd:.9f}",
                             *(f"{x:.3f}" for x in r[k]), *(f"{x:.6f}" for x in q)])
                oe = orbitalMotion.rv2elem(MU_EARTH, r[k], v[k])
                aw.writerow([i, spec["name"], f"{t[k]:.3f}", f"{jd:.9f}",
                             *(f"{x:.3f}" for x in r[k]), *(f"{x:.4f}" for x in v[k]),
                             *(f"{x:.6f}" for x in q), *(f"{x:.3e}" for x in wB[k]),
                             f"{oe.a:.3f}", f"{oe.e:.6f}", f"{oe.i * RAD2DEG:.4f}",
                             f"{oe.Omega * RAD2DEG:.4f}", f"{oe.omega * RAD2DEG:.4f}",
                             f"{oe.f * RAD2DEG:.4f}"])
            self.frames_logged = max(self.frames_logged, len(t))
        tf.close()
        af.close()

    # ---- rendering ---------------------------------------------------------
    def render(self, cadence="all", cameras="per_sat", **kw):
        from . import render as _render
        self.flush()
        return _render.render_run(self, cadence=cadence, cameras=cameras, **kw)

    # ---- lifecycle ---------------------------------------------------------
    def flush(self):
        self._write()

    def close(self):
        self._write()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
