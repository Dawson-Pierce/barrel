"""Generic sensor and onboard-estimation plumbing for BARREL scenes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class MeasurementBatch:
    """Measurements produced by one sensor acquisition."""

    sensor: str
    time_s: float
    data: Any
    frame: str
    source_ids: tuple = ()
    metadata: dict = field(default_factory=dict)


class Sensor:
    """Base class for a stateful, scheduled physical sensor model."""

    def __init__(self, name=None, cadence_s=0.0):
        self.name = name or type(self).__name__
        self.cadence_s = float(cadence_s)
        self.measurements: list[MeasurementBatch] = []
        self._last_acquisition_s = None

    def is_due(self, time_s):
        return (self._last_acquisition_s is None or self.cadence_s <= 0.0
                or float(time_s) + 1e-12 >= self._last_acquisition_s + self.cadence_s)

    def acquire(self, scene, host, time_s, rng):
        raise NotImplementedError

    def record(self, batch):
        self._last_acquisition_s = float(batch.time_s)
        self.measurements.append(batch)
        return batch

    @property
    def latest(self):
        return self.measurements[-1] if self.measurements else None

    def clear(self):
        self.measurements.clear()


class Estimator:
    """Small common interface implemented by Pierced/BREW estimator adapters."""

    def predict(self, time_s, dt_s, host=None):
        raise NotImplementedError

    def correct(self, measurements, time_s, host=None):
        raise NotImplementedError

    def estimate(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


@dataclass
class EstimatorNode:
    """One estimator and the data source used for its correction.

    ``source`` is ``"sensor"`` for raw measurements or the name of an upstream
    estimator.  The latter is how an intent estimator consumes pose estimates.
    """

    name: str
    estimator: Estimator
    source: str = "sensor"
    enabled: bool = True


class SensorChannel:
    """A physical sensor plus an ordered onboard estimator graph."""

    def __init__(self, sensor: Sensor, estimators: Iterable[EstimatorNode] = ()):
        self.sensor = sensor
        self.estimators = []
        for node in estimators:
            self.add_estimator(node)

    def add_estimator(self, node):
        if any(old.name == node.name for old in self.estimators):
            raise ValueError(f"duplicate estimator name {node.name!r}")
        if node.source != "sensor" and not any(old.name == node.source for old in self.estimators):
            raise ValueError(f"estimator {node.name!r} depends on unknown/upstream-late {node.source!r}")
        self.estimators.append(node)
        return node.estimator

    def acquire_and_update(self, scene, host, time_s, dt_s, rng):
        if not self.sensor.is_due(time_s):
            return None
        batch = self.sensor.acquire(scene, host, time_s, rng)
        outputs = {}
        for node in self.estimators:
            if not node.enabled:
                continue
            node.estimator.predict(time_s, dt_s, host=host)
            value = batch if node.source == "sensor" else outputs[node.source]
            node.estimator.correct(value, time_s, host=host)
            outputs[node.name] = node.estimator.estimate()
        return batch, outputs

