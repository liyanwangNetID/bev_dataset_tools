import numpy as np

from step6.camera_visibility import TARGET_WIDTH, project_voxel_contract, voxel_centers
from step6.contract import GRID, UNKNOWN_ID


def _camera():
    rotation = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=float)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    intrinsic = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]], dtype=float)
    valid = np.ones((540, 960), dtype=bool)
    return transform, intrinsic, valid


def test_open_ray_supervises_valid_frustum_to_grid_exit():
    semantics = np.full(GRID.shape, UNKNOWN_ID, dtype=np.uint8)
    transform, intrinsic, valid = _camera()
    frustum, free, supervised, stats = project_voxel_contract(
        semantics, transform, intrinsic, valid
    )
    assert np.any(free & supervised)
    assert np.array_equal(supervised, frustum)
    assert stats["surface_bounded_supervised"] == 0
    assert stats["grid_bounded_open_ray_supervised"] == int(frustum.sum())
    assert stats["open_ray_limit_policy"] == "occupancy_grid_exit"


def test_known_surface_pixel_has_no_supervision_behind_nearest_depth():
    semantics = np.full(GRID.shape, UNKNOWN_ID, dtype=np.uint8)
    surface_index = np.array([110, 100, 5], dtype=np.int64)
    semantics[tuple(surface_index)] = 0
    transform, intrinsic, valid = _camera()
    _, _, supervised, stats = project_voxel_contract(
        semantics, transform, intrinsic, valid
    )

    all_indices = np.indices(GRID.shape, dtype=np.int16).reshape(3, -1).T
    points = voxel_centers(all_indices)
    homogeneous = np.column_stack((points, np.ones(len(points))))
    camera = (homogeneous @ transform.T)[:, :3]
    positive = camera[:, 2] > 1e-6
    uvw = camera @ intrinsic.T
    px = np.rint(uvw[:, 0] / uvw[:, 2]).astype(np.int64)
    py = np.rint(uvw[:, 1] / uvw[:, 2]).astype(np.int64)

    surface_flat = np.ravel_multi_index(tuple(surface_index), GRID.shape)
    surface_pixel = py[surface_flat] * TARGET_WIDTH + px[surface_flat]
    same_pixel = positive & ((py * TARGET_WIDTH + px) == surface_pixel)
    behind = same_pixel & (camera[:, 2] > camera[surface_flat, 2] + GRID.voxel_size)
    assert np.any(behind)
    assert not np.any(supervised.reshape(-1)[behind])
    assert stats["surface_bounded_supervised"] > 0
