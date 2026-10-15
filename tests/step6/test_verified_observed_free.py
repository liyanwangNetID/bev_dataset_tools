import numpy as np

from step1.clip_manifest import CAMERA_NAMES
from step6.apply_camera_visibility import compute_semantics_and_mask
from step6.contract import GRID, OBSERVED_FREE_ID, UNKNOWN_ID


def _contracts():
    rotation = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=float)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    intrinsic = np.array([[1, 0, 50], [0, 1, 50], [0, 0, 1]], dtype=float)
    valid = np.ones((540, 960), dtype=bool)
    contract = (transform, intrinsic, valid, "asset")
    return {name: contract for name in CAMERA_NAMES}


def test_observed_free_is_limited_to_supervised_ray_before_surface():
    semantics = np.full(GRID.shape, UNKNOWN_ID, dtype=np.uint8)
    semantics[110, 100, 5] = 0

    updated, mask_camera, stats = compute_semantics_and_mask(semantics, _contracts())

    assert updated[110, 100, 5] == 0
    assert mask_camera[110, 100, 5]
    assert np.count_nonzero(updated == OBSERVED_FREE_ID) > 0
    assert np.all(mask_camera[updated == OBSERVED_FREE_ID])
    assert not np.any(mask_camera[updated == UNKNOWN_ID])
    assert all(value["verified_observed_free"] > 0 for value in stats.values())


def test_valid_frustum_without_known_surface_stays_unknown():
    semantics = np.full(GRID.shape, UNKNOWN_ID, dtype=np.uint8)

    updated, mask_camera, stats = compute_semantics_and_mask(semantics, _contracts())

    assert not np.any(updated == OBSERVED_FREE_ID)
    assert not mask_camera.any()
    assert all(value["verified_observed_free"] == 0 for value in stats.values())
