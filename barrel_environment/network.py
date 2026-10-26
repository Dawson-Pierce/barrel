"""Generic cross-observer estimate routing for BARREL constellations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from barrel_sensors import Estimator, MeasurementBatch


@dataclass(frozen=True)
class RingFusionProduct:
    """Estimate accumulated around one orbital ring and returned to its start node."""

    ring_id: int
    start_node: object
    route: tuple
    estimate: object
    time_s: float

    @property
    def plane(self):
        return self.ring_id

    @property
    def downlink_sat(self):
        return self.start_node

    @property
    def order(self):
        return self.route

    @property
    def mixture(self):
        return self.estimate


class SatelliteFusionRing:
    """Route a running fused estimate through one orbital plane.

    Satellites are supplied in increasing-slot order.  ``motion_direction=+1``
    means increasing slot index is along-track, so the next communication node
    is the trailing satellite at index ``-1``.  BARREL owns this routing only;
    the injected binary ``fusion`` operation is normally provided by BREW.
    """

    def __init__(self, ring_id, satellites: Iterable, fusion: Callable,
                 estimate_getter: Callable | None = None, motion_direction=1):
        self.ring_id = int(ring_id)
        self.satellites = tuple(satellites)
        if not self.satellites:
            raise ValueError("a fusion ring requires at least one satellite")
        self.fusion = fusion
        self.estimate_getter = estimate_getter or self._default_estimate
        self.motion_direction = int(motion_direction)
        if self.motion_direction not in (-1, 1):
            raise ValueError("motion_direction must be -1 or +1")

    @staticmethod
    def _default_estimate(satellite):
        estimate = getattr(satellite, "estimate", None)
        return estimate() if callable(estimate) else estimate

    @staticmethod
    def _identity(satellite):
        return getattr(satellite, "sat_id", satellite)

    def index(self, node):
        if isinstance(node, int) and 0 <= node < len(self.satellites):
            return int(node)
        for index, satellite in enumerate(self.satellites):
            if node is satellite or self._identity(satellite) == node:
                return index
        raise ValueError("start node is not a member of this fusion ring")

    def trailing_index(self, index):
        return (int(index)-self.motion_direction) % len(self.satellites)

    def route(self, start_node):
        start = self.index(start_node)
        step = -self.motion_direction
        return tuple((start+step*k) % len(self.satellites)
                     for k in range(len(self.satellites)))

    def fuse(self, start_node, time_s=0.0):
        order = self.route(start_node)
        running = self.estimate_getter(self.satellites[order[0]])
        for index in order[1:]:
            running = self.fusion(running, self.estimate_getter(self.satellites[index]))
        start = self.satellites[order[0]]
        return RingFusionProduct(
            self.ring_id, self._identity(start),
            tuple(self._identity(self.satellites[index]) for index in order),
            running, float(time_s))


class GroundStation:
    """Collect fused ring products and update an attached global estimator."""

    def __init__(self, estimator: Estimator, name="ground", cadence_s=0.0):
        self.name = str(name)
        self.estimator = estimator
        self.cadence_s = float(cadence_s)
        self._pending: dict[int, RingFusionProduct] = {}
        self._last_update_s = None
        self.history: list[MeasurementBatch] = []

    def receive(self, product: RingFusionProduct):
        if not isinstance(product, RingFusionProduct):
            raise TypeError("ground station accepts RingFusionProduct messages")
        self._pending[product.ring_id] = product
        return product

    def is_due(self, time_s):
        return (self._last_update_s is None or self.cadence_s <= 0.0 or
                float(time_s)+1e-12 >= self._last_update_s+self.cadence_s)

    def update(self, time_s, dt_s=None, metadata=None):
        if not self._pending or not self.is_due(time_s):
            return None
        products = tuple(self._pending[key] for key in sorted(self._pending))
        batch_metadata = {"routes": tuple(product.route for product in products)}
        if metadata:
            batch_metadata.update(metadata)
        batch = MeasurementBatch(
            sensor=self.name, time_s=float(time_s),
            data=tuple(product.estimate for product in products),
            frame="fused-estimate",
            source_ids=tuple(product.ring_id for product in products),
            metadata=batch_metadata,
        )
        elapsed = (float(dt_s) if dt_s is not None else
                   0.0 if self._last_update_s is None else float(time_s)-self._last_update_s)
        self.estimator.predict(float(time_s), elapsed, host=self)
        self.estimator.correct(batch, float(time_s), host=self)
        self._last_update_s = float(time_s)
        self._pending.clear()
        self.history.append(batch)
        return self.estimator.estimate()

    def reset(self):
        self._pending.clear(); self.history.clear(); self._last_update_s = None
        self.estimator.reset()
