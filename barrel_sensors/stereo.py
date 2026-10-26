"""Stereo-vision sensor reconstruction: turn an EarthScene.render_stereo() result plus a
depth (or disparity) map into a metric point cloud, via barrel_render.data. The stereo
imagery is produced by EarthScene.render_stereo (Blender); this is the image -> cloud half."""

import numpy as np

from barrel_render import data as bdata


def reconstruct_pointcloud(stereo_info, depth=None, disparity=None, side="L",
                           stride=2, max_depth=None):
    """(N, 3) world-frame point cloud (Blender units) from a render_stereo() result.
    Provide either a `depth` map (positive distance along the camera -Z) or a `disparity`
    map (pixels); intrinsics, baseline and the camera world pose come from `stereo_info`."""
    fx, fy, cx, cy = stereo_info["intrinsics"]
    if depth is None:
        if disparity is None:
            raise ValueError("provide depth or disparity")
        depth = bdata.disparity_to_depth(disparity, fx, stereo_info["baseline"])
    cam_to_world = np.asarray(stereo_info[f"pose_{side}"], dtype=float)
    return bdata.backproject_depth(depth, fx, fy, cx, cy, cam_to_world,
                                   stride=stride, max_depth=max_depth)
