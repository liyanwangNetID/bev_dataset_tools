import numpy as np
from step6.camera_visibility import project_known_voxels
from step6.contract import GRID, UNKNOWN_ID


def _contract(focal_length=100):
    rotation = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=float)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    intrinsic = np.array(
        [[focal_length, 0, 50], [0, focal_length, 50], [0, 0, 1]], dtype=float
    )
    return transform, intrinsic


def test_known_voxel_projects_and_unknown_does_not():
    semantics = np.full(GRID.shape, UNKNOWN_ID, dtype=np.uint8)
    semantics[110, 100, 5] = 4
    transform, intrinsic = _contract()
    valid = np.ones((540, 960), dtype=bool)
    mask, stats = project_known_voxels(semantics, transform, intrinsic, valid)
    assert mask[110, 100, 5]
    assert stats["known"] == 1
    assert stats["zbuffer_visible"] == 1


def test_rectification_invalid_pixel_rejects_voxel():
    semantics = np.full(GRID.shape, UNKNOWN_ID, dtype=np.uint8)
    semantics[110, 100, 5] = 4
    transform, intrinsic = _contract()
    valid = np.zeros((540, 960), dtype=bool)
    mask, stats = project_known_voxels(semantics, transform, intrinsic, valid)
    assert not mask.any()
    assert stats["valid_projection"] == 0


def test_zbuffer_keeps_nearest_known_voxel_as_first_surface():
    semantics = np.full(GRID.shape, UNKNOWN_ID, dtype=np.uint8)
    semantics[110, 100, 5] = 4
    semantics[120, 100, 5] = 10
    transform, intrinsic = _contract(focal_length=1)
    valid = np.ones((540, 960), dtype=bool)
    mask, stats = project_known_voxels(semantics, transform, intrinsic, valid)
    assert stats["zbuffer_visible"] == 1
    assert stats["supervised"] == int(mask.sum())
    assert mask[110, 100, 5]
    assert not mask[120, 100, 5]
    assert mask.sum() > 1
