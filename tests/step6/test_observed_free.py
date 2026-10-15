import numpy as np
from step6.camera_visibility import project_voxel_contract
from step6.contract import GRID, OBSERVED_FREE_ID, UNKNOWN_ID


def _camera():
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = np.array(
        [[0, 1, 0], [0, 0, -1], [1, 0, 0]], dtype=float
    )
    intrinsic = np.array(
        [[100, 0, 480], [0, 100, 270], [0, 0, 1]], dtype=float
    )
    valid = np.ones((540, 960), dtype=bool)
    return extrinsic, intrinsic, valid


def test_observed_free_preserves_occupied_and_marks_valid_frustum():
    semantics = np.full(GRID.shape, UNKNOWN_ID, dtype=np.uint8)
    semantics[120, 100, 4] = 4
    extrinsic, intrinsic, valid = _camera()
    frustum, free, supervised, stats = project_voxel_contract(
        semantics, extrinsic, intrinsic, valid
    )
    updated = semantics.copy()
    updated[free] = OBSERVED_FREE_ID
    assert updated[120, 100, 4] == 4
    assert stats["valid_frustum"] > 0
    assert np.count_nonzero(updated == OBSERVED_FREE_ID) > 0
    assert np.count_nonzero(supervised) > 0
    assert np.all(frustum[supervised])


def test_rays_without_known_surface_are_not_supervised():
    semantics = np.full(GRID.shape, UNKNOWN_ID, dtype=np.uint8)
    extrinsic, intrinsic, valid = _camera()
    frustum, free, supervised, stats = project_voxel_contract(
        semantics, extrinsic, intrinsic, valid
    )
    assert stats["valid_frustum"] > 0
    assert np.count_nonzero(free) > 0
    assert np.count_nonzero(supervised) == 0
