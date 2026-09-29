import numpy as np
from step6.camera_visibility import project_known_voxels
from step6.contract import GRID, UNKNOWN_ID


def test_known_voxel_projects_and_unknown_does_not():
    semantics = np.full(GRID.shape, UNKNOWN_ID, dtype=np.uint8)
    semantics[110, 100, 5] = 4
    # Optical camera looking along ego +x: camera z=ego x, camera x=-ego y,
    # camera y=-ego z. Translation is zero for this synthetic contract test.
    rotation = np.array([[0,-1,0],[0,0,-1],[1,0,0]], dtype=float)
    transform = np.eye(4)
    transform[:3,:3] = rotation
    k = np.array([[100,0,50],[0,100,50],[0,0,1]], dtype=float)
    valid = np.ones((540,960), dtype=bool)
    mask, stats = project_known_voxels(semantics, transform, k, valid)
    assert mask.sum() == 1
    assert mask[110,100,5]
    assert stats["known"] == 1


def test_rectification_invalid_pixel_rejects_voxel():
    semantics = np.full(GRID.shape, UNKNOWN_ID, dtype=np.uint8)
    semantics[110,100,5] = 4
    rotation = np.array([[0,-1,0],[0,0,-1],[1,0,0]], dtype=float)
    transform = np.eye(4)
    transform[:3,:3] = rotation
    k = np.array([[100,0,50],[0,100,50],[0,0,1]], dtype=float)
    valid = np.zeros((540,960), dtype=bool)
    mask, stats = project_known_voxels(semantics, transform, k, valid)
    assert not mask.any()
    assert stats["valid_projection"] == 0


def test_zbuffer_keeps_nearest_known_voxel_per_pixel():
    semantics = np.full(GRID.shape, UNKNOWN_ID, dtype=np.uint8)
    semantics[110,100,5] = 4
    semantics[120,100,5] = 10
    rotation = np.array([[0,-1,0],[0,0,-1],[1,0,0]], dtype=float)
    transform = np.eye(4)
    transform[:3,:3] = rotation
    k = np.array([[1,0,50],[0,1,50],[0,0,1]], dtype=float)
    valid = np.ones((540,960), dtype=bool)
    mask, stats = project_known_voxels(semantics, transform, k, valid)
    assert mask.sum() == 1
    assert mask[110,100,5]
    assert stats["zbuffer_visible"] == 1
