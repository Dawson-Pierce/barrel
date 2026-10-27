"""One simulation owner for dynamics, spacecraft, sensing, and scheduled work."""

from dataclasses import dataclass, field
import heapq
import numpy as np

from . import coordinates


@dataclass
class Satellite:
    """A spacecraft and its payloads, with optional precomputed state history."""

    sat_id: int
    name: str
    plane: int = 0
    slot: int = 0
    t: np.ndarray | None = None
    jd: np.ndarray | None = None
    pos: np.ndarray | None = None
    quat: np.ndarray | None = None
    velocity: np.ndarray | None = None
    dynamics_id: int | None = None
    payloads: list = field(default_factory=list)
    position_eci: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity_eci: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation_eci: np.ndarray = field(default_factory=lambda: np.eye(3))
    epoch_jd: float = 0.0

    def add_payload(self, channel):
        self.payloads.append(channel)
        return channel

    def sensor_rotation_eci(self, sensor):
        return self.rotation_eci @ np.asarray(getattr(sensor, "mount_dcm", np.eye(3)))


class Simulation:
    """Own the clock and execute sources, sensors, estimator chains, and events.

    Spacecraft may use Basilisk dynamics or sampled ephemerides. Events execute
    after sensor acquisition, in registration order at equal times. A record
    callback runs last at each requested sample and can process study results.
    """

    def __init__(self, *, seed=0, name=None, output_root="output", log_dt=1.0,
                 epoch=(2026, 1, 1, 0, 0, 0), control_dt=1.0,
                 forces=("j2", "drag", "sun", "moon"), gravity_degree=32,
                 gravity_file=None):
        self.time_s = 0.0
        self.frame = 0
        self.satellites = []
        self.sources = []
        self.rng = np.random.default_rng(seed)
        self.events = {}
        self._queue = []
        self._event_order = 0
        self._emissions = None
        self._dynamics = None
        self._log_dt = float(log_dt)
        self._dynamics_options = dict(name=name, output_root=output_root, epoch=epoch,
                                     control_dt=control_dt, forces=forces,
                                     gravity_degree=gravity_degree, gravity_file=gravity_file)

    def add_satellite(self, *, name=None, elements=None, state=None, plane=0, slot=0,
                      **physical):
        sat = Satellite(len(self.satellites), name or f"sat{len(self.satellites)}", plane, slot)
        if elements is not None or state is not None:
            if self._dynamics is None:
                from ._dynamics import _BasiliskDynamics
                self._dynamics = _BasiliskDynamics(**self._dynamics_options)
            sat.dynamics_id = self._dynamics.add_satellite(
                name=sat.name, elements=elements, state=state, **physical)
        elif physical:
            raise ValueError("physical properties require elements or an initial state")
        self.satellites.append(sat)
        return sat

    def add_ephemeris(self, t, jd, pos, quat=None, *, name=None, plane=0, slot=0):
        t, jd, pos = np.asarray(t, dtype=float), np.asarray(jd, dtype=float), np.asarray(pos, dtype=float)
        if t.ndim != 1 or not len(t) or jd.shape != t.shape or pos.shape != (len(t), 3):
            raise ValueError("ephemeris needs times, Julian dates, and one 3D position per time")
        if np.any(np.diff(t) <= 0) or not all(np.all(np.isfinite(x)) for x in (t, jd, pos)):
            raise ValueError("ephemeris must be finite with strictly increasing times")
        quat = np.asarray(coordinates.nadir_attitude(pos, t) if quat is None else quat, dtype=float)
        if quat.shape != (len(t), 4) or not np.all(np.isfinite(quat)):
            raise ValueError("ephemeris needs one finite xyzw quaternion per time")
        sat = self.add_satellite(name=name, plane=plane, slot=slot)
        sat.t, sat.jd, sat.pos, sat.quat = t, jd, pos, quat
        sat.velocity = np.gradient(pos, t, axis=0) if len(t) > 1 else np.zeros_like(pos)
        return sat

    def add_source(self, source):
        """Register an object implementing emit(time_s)."""
        self.sources.append(source)
        return source

    def observed(self, name):
        return next(source.quantity for source in self.sources
                    if hasattr(source, "quantity") and source.quantity.name == name)

    def truth_frame(self, name, epoch_min):
        return self.observed(name).truth_frame(epoch_min)

    def emissions(self, kind=None):
        if self._emissions is None:
            self._emissions = []
            for source in self.sources:
                value = source.emit(self.time_s)
                if value is not None:
                    self._emissions.extend(value if isinstance(value, (list, tuple)) else [value])
        return [value for value in self._emissions if kind is None or value.kind == kind]

    def schedule(self, name, times, action):
        """Call action(simulation) at each time; expose its latest return in events[name]."""
        for time in np.atleast_1d(times):
            if not np.isfinite(time) or time < self.time_s:
                raise ValueError("scheduled times must be finite and cannot be in the past")
            heapq.heappush(self._queue, (float(time), self._event_order, name, action))
            self._event_order += 1

    def _satellite(self, satellite):
        if isinstance(satellite, Satellite):
            return satellite
        if isinstance(satellite, str):
            return next(sat for sat in self.satellites if sat.name == satellite)
        return self.satellites[int(satellite)]

    def state(self, satellite):
        sat = self._satellite(satellite)
        if sat.dynamics_id is not None:
            return self._dynamics.state(sat.dynamics_id)
        return dict(position=sat.position_eci, velocity=sat.velocity_eci,
                    rotation=sat.rotation_eci, jd=sat.epoch_jd, time_s=self.time_s)

    def command(self, satellite, **command):
        sat = self._satellite(satellite)
        if sat.dynamics_id is None:
            raise ValueError("commands require a dynamically propagated spacecraft")
        self._dynamics.command(sat.dynamics_id, **command)

    def _advance(self, time_s, acquire):
        dt = float(time_s) - self.time_s
        if self._dynamics is not None and (dt > 0 or not self._dynamics._built):
            self._dynamics.run(dt, dt=self._log_dt)
        self.time_s = float(time_s)
        self._emissions = None
        for sat in self.satellites:
            if sat.dynamics_id is not None:
                state = self._dynamics.state(sat.dynamics_id)
                sat.position_eci, sat.velocity_eci = state["position"], state["velocity"]
                sat.rotation_eci, sat.epoch_jd = state["rotation"], state["jd"]
            elif sat.t is not None:
                right = min(int(np.searchsorted(sat.t, self.time_s)), len(sat.t) - 1)
                left = max(right - 1, 0)
                k = left if abs(sat.t[left] - self.time_s) <= abs(sat.t[right] - self.time_s) else right
                j = min(k + 1, len(sat.t) - 1) if sat.t[k] <= self.time_s else max(k - 1, 0)
                alpha = np.clip((self.time_s - sat.t[k]) / (sat.t[j] - sat.t[k]), 0, 1) if j != k else 0
                sat.position_eci = (1 - alpha) * sat.pos[k] + alpha * sat.pos[j]
                sat.velocity_eci = sat.velocity[k]
                sat.rotation_eci = coordinates.quat_to_dcm(sat.quat[k])
                sat.epoch_jd = float(sat.jd[k])
        if acquire:
            for sat in self.satellites:
                for channel in sat.payloads:
                    channel.acquire_and_update(self, sat, self.time_s, dt, self.rng)

    def step_to(self, time_s, *, acquire=True):
        target = float(time_s)
        if not np.isfinite(target) or target < self.time_s:
            raise ValueError("simulation time must be finite and cannot move backward")
        self.events = {}
        while self._queue and self._queue[0][0] < target:
            event_time = self._queue[0][0]
            self._advance(event_time, acquire=False)
            self._execute_events(event_time)
        self._advance(target, acquire)
        self._execute_events(target)
        return self

    def _execute_events(self, time_s):
        while self._queue and self._queue[0][0] <= time_s:
            _, _, name, action = heapq.heappop(self._queue)
            self.events[name] = action(self)

    def run(self, times, *, record=None, acquire=True):
        """Run samples in order; record(simulation) processes each completed sample."""
        for self.frame, time_s in enumerate(times):
            self.step_to(time_s, acquire=acquire)
            if record is not None:
                record(self)

    def close(self):
        if self._dynamics is not None:
            self._dynamics.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
