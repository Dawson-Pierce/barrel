"""barrel_sensors: Python sensor/observation models for the BARREL constellation.

The C++ Mission core is tracker-agnostic and the EO sensors were trimmed from it; these
pure-Python models bring them back (not speed-critical) on top of the logged trajectory:

    CmosSensor   - a nadir CMOS imager: geolocate ground points, quantize to a pixel,
                   remap back to ECEF/geodetic for a global tracking frame.
    reconstruct_pointcloud - turn an EarthScene.render_stereo() result + depth into a
                   metric point cloud (stereo vision sensor) via barrel_render.data.

The stereo *imaging* itself lives in barrel_render.EarthScene.render_stereo (Blender side).
"""

from .base import Estimator, EstimatorNode, MeasurementBatch, Sensor, SensorChannel
from .cmos import CmosSensor
from .lidar import LidarSensor
from .stereo import reconstruct_pointcloud

__all__ = [
    "CmosSensor", "Estimator", "EstimatorNode", "LidarSensor", "MeasurementBatch", "Sensor",
    "SensorChannel", "reconstruct_pointcloud",
]
