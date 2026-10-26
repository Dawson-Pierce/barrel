"""Geometric lidar returns from the first mesh intersection of each angular beam."""

from itertools import product
import numpy as np


class LidarMesh:
    """Body-frame triangle geometry and its reusable ray-intersection structure."""

    def __init__(self, triangles):
        import trimesh
        from trimesh.ray.ray_pyembree import RayMeshIntersector

        triangles = np.asarray(triangles, dtype=float)
        self.mesh = trimesh.Trimesh(
            vertices=triangles.reshape(-1, 3),
            faces=np.arange(triangles.size // 3).reshape(-1, 3),
            process=False)
        self.rays = RayMeshIntersector(self.mesh)
        self.corners = np.array(list(product(*self.mesh.bounds.T)))


def sensor_frame(boresight):
    """Columns are forward, horizontal, and vertical axes in observer LVLH."""
    forward = np.asarray(boresight, dtype=float)
    forward = forward / np.linalg.norm(forward)
    up = np.array([0., 0., 1.])
    if abs(forward @ up) > .99:
        up = np.array([0., 1., 0.])
    horizontal = np.cross(up, forward)
    horizontal /= np.linalg.norm(horizontal)
    return np.column_stack([forward, horizontal, np.cross(forward, horizontal)])


def angular_beams(pixels=256, fov_deg=20.):
    """One ray through the centre of each angular cell of the square field of view."""
    pitch = np.deg2rad(fov_deg) / pixels
    angles = (np.arange(pixels) + .5 - pixels / 2) * pitch
    horizontal, vertical = np.meshgrid(np.tan(angles), np.tan(angles))
    directions = np.column_stack([np.ones(pixels * pixels),
                                  horizontal.ravel(), vertical.ravel()])
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


def raycast_scan(meshes, poses, boresight, pixels=256, fov_deg=20.):
    """Return clean point clouds, including self-occlusion and inter-object blocking.

    The observer is at the LVLH origin. Each pose contains a body-origin position
    and body-to-LVLH rotation. Every beam contributes at most its nearest hit.
    Empty clouds represent objects outside the viewing cone or fully occluded.
    """
    directions = angular_beams(pixels, fov_deg) @ sensor_frame(boresight).T
    return _first_returns(meshes, poses, directions)


def surround_scan(meshes, poses, angular_pitch=np.deg2rad(20.) / 256):
    """Ideal full-surround array sampled on six nonoverlapping angular faces.

    The heads share the observer origin. Mesh bounds restrict ray construction
    to potentially occupied pixels, without changing the full angular grid.
    The first-intersection calculation still includes every object for each ray.
    """
    pixels = int(round((np.pi / 2) / angular_pitch))
    pitch = (np.pi / 2) / pixels
    directions = []
    world_bounds = [mesh.corners @ np.asarray(pose[1]).T + pose[0]
                    for mesh, pose in zip(meshes, poses)]
    for forward in np.vstack([np.eye(3), -np.eye(3)]):
        frame = sensor_frame(forward)
        candidates = []
        for bounds in world_bounds:
            camera = bounds @ frame
            depth, horizontal, vertical = camera.T
            planes = np.column_stack([depth, depth - horizontal, depth + horizontal,
                                      depth - vertical, depth + vertical])
            if np.any(planes.max(axis=0) <= 0.):
                continue
            if np.all(depth > 0.):
                angles = np.arctan(camera[:, 1:] / depth[:, None])
                lower = np.floor((angles.min(0) + np.pi / 4) / pitch - .5).astype(int)
                upper = np.ceil((angles.max(0) + np.pi / 4) / pitch - .5).astype(int)
                lower, upper = np.maximum(lower, 0), np.minimum(upper, pixels - 1)
            else:
                lower, upper = np.zeros(2, dtype=int), np.full(2, pixels - 1)
            columns, rows = np.meshgrid(np.arange(lower[0], upper[0] + 1),
                                       np.arange(lower[1], upper[1] + 1))
            candidates.append((rows * pixels + columns).ravel())
        if candidates:
            ids = np.unique(np.concatenate(candidates))
            horizontal = np.tan((ids % pixels + .5 - pixels / 2) * pitch)
            vertical = np.tan((ids // pixels + .5 - pixels / 2) * pitch)
            rays = np.column_stack([np.ones(len(ids)), horizontal, vertical])
            rays /= np.linalg.norm(rays, axis=1, keepdims=True)
            directions.append(rays @ frame.T)
    if not directions:
        return [np.zeros((3, 0)) for _ in meshes]
    return _first_returns(meshes, poses, np.vstack(directions))


def _first_returns(meshes, poses, directions):
    distances = np.full(len(directions), np.inf)
    owner = np.full(len(directions), -1, dtype=int)
    hits = np.zeros_like(directions)
    for index, (mesh, pose) in enumerate(zip(meshes, poses)):
        position, rotation = pose[:2]
        position, rotation = np.asarray(position), np.asarray(rotation)
        origin = -rotation.T @ position
        locations, beam_ids, _ = mesh.rays.intersects_location(
            np.broadcast_to(origin, directions.shape), directions @ rotation,
            multiple_hits=False)
        world = locations @ rotation.T + position
        ranges = np.linalg.norm(world, axis=1)
        nearer = ranges < distances[beam_ids]
        beam_ids = beam_ids[nearer]
        distances[beam_ids] = ranges[nearer]
        owner[beam_ids] = index
        hits[beam_ids] = world[nearer]
    return [hits[owner == index].T.copy() for index in range(len(meshes))]


def noisy_returns(clouds, rng, return_cap=0, point_noise=.01, position_noise=.02):
    """Apply an optional point budget, independent noise, and one offset per object."""
    result = []
    for cloud in clouds:
        if return_cap and cloud.shape[1] > return_cap:
            cloud = cloud[:, rng.choice(cloud.shape[1], return_cap, replace=False)]
        if cloud.shape[1]:
            cloud = cloud + rng.normal(0., point_noise, cloud.shape)
            cloud = cloud + rng.normal(0., position_noise, (3, 1))
        result.append(cloud)
    return result
