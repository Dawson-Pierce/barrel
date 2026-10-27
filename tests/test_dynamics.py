import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from Basilisk.architecture import messaging

from barrel_simulation._dynamics import _BasiliskDynamics
from barrel_simulation import _dynamics as mission_module
from barrel_simulation.constellation import interpolated_states, walker_elements


class DynamicsInitializationTests(unittest.TestCase):
    def test_drag_density_is_read_before_initial_force_evaluation(self):
        factory = mission_module.dragDynamicEffector.DragDynamicEffector

        def stale_drag():
            drag = factory()
            # Make the unread native cache deterministically invalid instead of
            # relying on whatever bytes the allocator happens to return.
            payload = messaging.AtmoPropsMsgPayload()
            payload.neutralDensity = float("nan")
            message = messaging.AtmoPropsMsg().write(payload)
            drag.atmoDensInMsg.subscribeTo(message)
            drag.ReadInputs()
            return drag

        for wheels, step in ((False, 10.0), (True, 1.0)):
            with self.subTest(wheels=wheels), tempfile.TemporaryDirectory() as output:
                with patch.object(mission_module.dragDynamicEffector,
                                  "DragDynamicEffector", side_effect=stale_drag):
                    mission = _BasiliskDynamics(output_root=output, control_dt=step)
                    for element in walker_elements(2, 2):
                        mission.add_satellite(elements=element, wheels=wheels)
                    mission.run(duration=20.0, dt=step)
                positions, velocities = interpolated_states(mission, np.arange(21.0))
                self.assertTrue(np.all(np.isfinite(positions)))
                self.assertTrue(np.all(np.isfinite(velocities)))
                for index, spec in enumerate(mission._specs):
                    position, velocity = mission._initial_rv(spec)
                    np.testing.assert_allclose(positions[0, index], position, rtol=0, atol=1e-8)
                    np.testing.assert_allclose(velocities[0, index], velocity, rtol=0, atol=1e-10)
                    self.assertTrue(np.all(np.isfinite(mission._recs[index].sigma_BN)))


if __name__ == "__main__":
    unittest.main()
