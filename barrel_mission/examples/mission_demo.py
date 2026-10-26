"""Mission demo: a Walker-like constellation at 550 km, propagated for one orbit with the
Basilisk dynamics, a pointing command, and an impulsive maneuver, logged to CSV and rendered.

Runs inside the barrel image (Python + Basilisk + Blender), with the repo mounted at /work:

  docker run --rm -v "<repo>:/work" -w /work barrel \
    python3 barrel_mission/examples/mission_demo.py

Output -> output/<timestamp>/{trajectory.csv, analysis.csv, renders/...}
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from barrel_mission import Mission
from barrel_mission.mission import MU_EARTH, R_EARTH, DEG2RAD

A = R_EARTH + 550e3
N = 4
PERIOD = 2 * math.pi * math.sqrt(A ** 3 / MU_EARTH)

m = Mission()                          # default full physics: SH gravity + drag + Sun/Moon + RWs
for k in range(N):
    m.add_satellite(
        elements=(A, 0.001, 53 * DEG2RAD, (360.0 * k / N) * DEG2RAD, 0.0,
                  (360.0 * ((k * 7) % N) / N) * DEG2RAD),
        name=f"sat{k}", pointing="nadir",
        cameras=[("overview",), ("sensor", 30.0, 55.0)])

m.run(duration=PERIOD * 0.25, dt=30.0)                   # quarter orbit
m.command("sat0", pointing="target", target_latlon=(0.0, 0.0))   # task sat0 over (0,0)
m.run(duration=PERIOD * 0.25, dt=30.0)
m.command("sat2", delta_v_eci=(0.0, 5.0, 0.0))           # 5 m/s along-track burn
m.run(duration=PERIOD * 0.5, dt=30.0)
m.close()
print(f"[demo] logged {m.frames_logged} samples over 1 orbit (~{PERIOD / 60:.1f} min)")

m.render(cadence=40, cameras="per_sat")                  # ~5 frames: overview + per-sat EO
print(f"[demo] done -> {m.run_dir}")
