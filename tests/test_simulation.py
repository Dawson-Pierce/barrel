import tempfile
import unittest
import numpy as np

from barrel_simulation import Simulation, WalkerDeltaConstellation
from barrel_sensors import Estimator, EstimatorNode, Sensor, SensorChannel


class Recorder(Estimator):
    def __init__(self, convert):
        self.convert = convert
        self.steps = []

    def predict(self, time_s, dt_s, host=None):
        self.steps.append((time_s, dt_s))

    def correct(self, value, time_s, host=None):
        self.value = self.convert(value)

    def estimate(self):
        return self.value


class SimulationTests(unittest.TestCase):
    def test_chain_branching_and_sensor_cadence_with_intermediate_events(self):
        sim = Simulation()
        first = Recorder(lambda batch: batch.data)
        second = Recorder(lambda value: value + 10)
        branch = Recorder(lambda value: value * 2)
        channel = SensorChannel(Sensor("test", cadence_s=2, sample=lambda s, h, t, r: t), [
            EstimatorNode("first", first), EstimatorNode("second", second, "first"),
            EstimatorNode("branch", branch, "first"),
        ])
        sim.add_satellite().add_payload(channel)
        sim.schedule("command", [0.5, 1.5], lambda s: s.time_s)
        sim.run([0, 1, 2, 3, 4])
        self.assertEqual(first.steps, [(0, 0), (2, 2), (4, 2)])
        self.assertEqual(channel.outputs, {"first": 4, "second": 14, "branch": 8})

    def test_event_order_and_recording_after_acquisition(self):
        sim, log = Simulation(), []
        sensor = Sensor(sample=lambda s, h, t, r: log.append((t, "sensor")))
        sim.add_satellite().add_payload(SensorChannel(sensor))
        sim.schedule("first", [0, 2], lambda s: log.append((s.time_s, "first")))
        sim.schedule("second", [0, 2], lambda s: log.append((s.time_s, "second")))
        sim.run([0, 2], record=lambda s: log.append((s.time_s, "record")))
        self.assertEqual(log, [(t, stage) for t in (0, 2)
                              for stage in ("sensor", "first", "second", "record")])
        with self.assertRaises(ValueError):
            sim.step_to(1)

    def test_ephemeris_and_live_state_use_one_spacecraft_object(self):
        sim = Simulation()
        sat = sim.add_ephemeris([0, 10], [2451545, 2451545 + 10 / 86400],
                                [[7000000, 0, 0], [7000000, 75000, 0]])
        sim.step_to(5)
        np.testing.assert_array_equal(sim.state(sat)["position"], [7000000, 37500, 0])
        np.testing.assert_array_equal(sat.velocity_eci, [0, 7500, 0])
        with self.assertRaises(ValueError):
            sim.command(sat, delta_v_eci=[0, 1, 0])

    def test_basilisk_state_and_command_through_public_simulation(self):
        with tempfile.TemporaryDirectory() as output:
            with Simulation(output_root=output, log_dt=1) as sim:
                sat = sim.add_satellite(elements=WalkerDeltaConstellation(1, 1).elements()[0])
                sim.run([0, 1, 2])
                before = sim.state(sat)["velocity"].copy()
                sim.command(sat, delta_v_eci=[0, 0.1, 0])
                sim.run([3, 4])
                self.assertTrue(np.all(np.isfinite(sim.state(sat)["rotation"])))
                self.assertGreater(np.linalg.norm(sim.state(sat)["velocity"] - before), 0)
                self.assertEqual(sim.state(sat)["time_s"], 4)

    def test_uncached_parallel_constellation_and_cache_reuse(self):
        with tempfile.TemporaryDirectory() as cache:
            shell = WalkerDeltaConstellation(1, 2)
            first = shell.propagate(minutes=0.5, dt_s=1, workers=2, cache_root=cache)
            second = shell.propagate(minutes=0.5, dt_s=1, workers=1, cache_root=cache)
            self.assertEqual(first["pos"].shape, (31, 2, 3))
            self.assertTrue(np.all(np.isfinite(first["pos"])))
            np.testing.assert_array_equal(first["pos"], second["pos"])


if __name__ == "__main__":
    unittest.main()
