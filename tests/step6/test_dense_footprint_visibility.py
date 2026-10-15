import numpy as np

from step6.camera_visibility import project_voxel_contract
from step6.contract import GRID, UNKNOWN_ID


def _camera():
    rotation = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=float)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    intrinsic = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]], dtype=float)
    valid = np.ones((540, 960), dtype=bool)
    return transform, intrinsic, valid


def test_occupied_footprint_is_wider_than_center_pixel_and_preserves_occlusion():
    semantics = np.full(GRID.shape, UNKNOWN_ID, dtype=np.uint8)
    semantics[110, 100, 5] = 0
    transform, intrinsic, valid = _camera()

    frustum, free, supervised, stats = project_voxel_contract(
        semantics, transform, intrinsic, valid
    )

    assert stats["first_surface"] == 1
    assert stats["footprint_surface_pixels"] > stats["first_surface"]
    assert np.count_nonzero(free & supervised) > 1
    assert supervised[110, 100, 5]
    assert not supervised[120, 100, 5]


def test_invalid_rectified_pixels_do_not_receive_footprint_depth():
    semantics = np.full(GRID.shape, UNKNOWN_ID, dtype=np.uint8)
    semantics[110, 100, 5] = 0
    transform, intrinsic, valid = _camera()
    valid[:] = False
    _, _, supervised, stats = project_voxel_contract(semantics, transform, intrinsic, valid)
    assert not supervised.any()
    assert stats["footprint_surface_pixels"] == 0
