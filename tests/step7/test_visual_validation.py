import numpy as np
from bev_dataset_tools.step7.visualization import height_slices, mask_bev, occupancy_bev


def test_visual_products_have_expected_shapes():
    semantics = np.full((200, 200, 16), 17, dtype=np.uint8)
    semantics[20:30, 40:50, 3] = 4
    mask = np.zeros_like(semantics, dtype=bool)
    mask[10:100, 10:100, :] = True
    assert occupancy_bev(semantics, mask).shape == (800, 800, 3)
    assert mask_bev(mask).shape == (800, 800, 3)
    assert height_slices(semantics, mask).shape == (960, 960, 3)
