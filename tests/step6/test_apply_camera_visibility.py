import numpy as np
from step6.apply_camera_visibility import compute_semantics_and_mask
from step6.contract import GRID, UNKNOWN_ID


def test_compute_semantics_and_mask_unions_camera_masks():
    semantics = np.full(GRID.shape, UNKNOWN_ID, dtype=np.uint8)
    semantics[110, 100, 5] = 4
    rotation = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=float)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    k_rect = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]], dtype=float)
    valid_mask = np.ones((540, 960), dtype=bool)
    contract = (transform, k_rect, valid_mask, "asset")
    contracts = {
        "cross_left": contract,
        "front_wide": contract,
        "cross_right": contract,
        "front_tele": contract,
    }

    updated, mask_camera, stats = compute_semantics_and_mask(semantics, contracts)

    assert updated.shape == GRID.shape
    assert mask_camera.sum() > 1
    assert all(value["first_surface"] == 1 for value in stats.values())
