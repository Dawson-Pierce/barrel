import unittest

import numpy as np

from barrel_environment import (
    DetectionEmission, GroundStation, Satellite, SatelliteFusionRing, Scene,
    SatelliteSurfaceEmitter,
)
from barrel_sensors import Estimator, EstimatorNode, LidarSensor, MeasurementBatch, Sensor, SensorChannel


class PointSensor(Sensor):
    def acquire(self, scene, host, time_s, rng):
        values = [e.data for e in scene.emissions("test")]
        return self.record(MeasurementBatch(self.name, time_s, values, "test"))


class EchoEstimator(Estimator):
    def __init__(self, transform=lambda x: x):
        self.transform, self.value = transform, None

    def predict(self, time_s, dt_s, host=None):
        pass

    def correct(self, measurements, time_s, host=None):
        self.value = self.transform(measurements)

    def estimate(self):
        return self.value

    def reset(self):
        self.value = None


class CountingEmitter:
    def __init__(self):
        self.calls = 0

    def emit(self, time_s):
        self.calls += 1
        return DetectionEmission("source", "test", "test", self.calls)


class RuntimeTests(unittest.TestCase):
    def test_fusion_ring_routes_toward_trailing_satellite(self):
        satellites = [type("Node", (), {"sat_id": k, "estimate": k})() for k in range(4)]
        ring = SatelliteFusionRing(2, satellites, lambda running, value: running+value,
                                   estimate_getter=lambda sat: (sat.estimate,))
        product = ring.fuse(start_node=1, time_s=12.0)
        self.assertEqual(product.route, (1, 0, 3, 2))
        self.assertEqual(product.estimate, (1, 0, 3, 2))
        self.assertEqual(product.start_node, 1)

    def test_ground_station_corrects_estimator_with_all_ring_products(self):
        estimator = EchoEstimator(lambda batch: (batch.source_ids, batch.data))
        ground = GroundStation(estimator)
        nodes = [type("Node", (), {"sat_id": k, "estimate": k})() for k in range(2)]
        first = SatelliteFusionRing(0, nodes, lambda a, b: a+b,
                                    estimate_getter=lambda sat: sat.estimate).fuse(0, 10.0)
        second = SatelliteFusionRing(1, nodes, lambda a, b: a+b,
                                     estimate_getter=lambda sat: sat.estimate).fuse(1, 10.0)
        ground.receive(second); ground.receive(first)
        source_ids, estimates = ground.update(10.0, 2.0)
        self.assertEqual(source_ids, (0, 1))
        self.assertEqual(estimates, (1, 1))

    def test_scene_routes_registered_ring_to_ground_on_command(self):
        estimator = EchoEstimator(lambda batch: batch.data)
        ground = GroundStation(estimator)
        nodes = [type("Node", (), {"sat_id": k, "estimate": k+1})() for k in range(3)]
        ring = SatelliteFusionRing(0, nodes, lambda a, b: a+b,
                                   estimate_getter=lambda sat: sat.estimate)
        runtime = Scene(); runtime.add_fusion_ring(ring); runtime.add_ground_station(ground)
        product = runtime.route_ring(ring, start_node=0, ground_station=ground, time_s=5.0)
        self.assertEqual(product.estimate, 6)
        self.assertEqual(ground.update(5.0, 1.0), (6,))

    def test_epoch_emission_is_shared_by_sensors(self):
        scene, emitter = Scene(seed=1), CountingEmitter()
        scene.add_detectable(emitter)
        a, b = Satellite(0), Satellite(1)
        a.add_payload(SensorChannel(PointSensor("a")))
        b.add_payload(SensorChannel(PointSensor("b")))
        scene.add_satellite(a); scene.add_satellite(b)
        scene.step_to(1.0)
        self.assertEqual(emitter.calls, 1)
        self.assertEqual(a.payloads[0].sensor.latest.data, b.payloads[0].sensor.latest.data)

    def test_estimator_dependency_uses_upstream_output(self):
        raw = EchoEstimator(lambda batch: batch.data[0])
        downstream = EchoEstimator(lambda value: value + 10)
        channel = SensorChannel(PointSensor("sensor"), [
            EstimatorNode("pose", raw), EstimatorNode("intent", downstream, source="pose")
        ])
        scene, emitter, sat = Scene(seed=1), CountingEmitter(), Satellite(0)
        scene.add_detectable(emitter); sat.add_payload(channel); scene.add_satellite(sat)
        scene.step_to(1.0)
        self.assertEqual(downstream.estimate(), 11)

    def test_lidar_returns_front_surface_cloud(self):
        scene, sat = Scene(seed=1), Satellite(0)
        sat.position_eci = np.zeros(3)
        sat.add_payload(SensorChannel(LidarSensor(max_range_m=100.0, fov_deg=30.0,
                                                  range_sigma_m=0.0, cross_range_sigma_m=0.0)))
        surface = np.array([[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])
        state = {"position": np.array([0.0, 0.0, 10.0]), "rotation": np.eye(3)}
        scene.add_detectable(SatelliteSurfaceEmitter(1, state, surface))
        scene.add_satellite(sat); scene.step_to(1.0)
        self.assertGreater(sat.payloads[0].sensor.latest.data.shape[1], 0)


if __name__ == "__main__":
    unittest.main()
