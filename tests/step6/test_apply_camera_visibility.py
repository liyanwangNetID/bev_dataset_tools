import numpy as np
from step6.apply_camera_visibility import compute_mask
from step6.contract import GRID, UNKNOWN_ID


def test_compute_mask_unions_camera_masks():
    semantics = np.full(GRID.shape, UNKNOWN_ID, dtype=np.uint8)
    semantics[110,100,5] = 4
    rotation = np.array([[0,-1,0],[0,0,-1],[1,0,0]], dtype=float)
    transform = np.eye(4)
    transform[:3,:3] = rotation
    k = np.array([[100,0,50],[0,100,50],[0,0,1]], dtype=float)
    valid = np.ones((540,960), dtype=bool)
    contract = (transform, k, valid, "asset")
    contracts = {
        "cross_left": contract,
        "front_wide": contract,
        "cross_right": contract,
        "front_tele": contract,
    }
    mask, stats = compute_mask(semantics, contracts)
    assert mask.sum() == 1
    assert all(value["zbuffer_visible"] == 1 for value in stats.values())
