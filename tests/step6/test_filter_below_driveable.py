import numpy as np
from step6.contract import OBSERVED_FREE_ID, UNKNOWN_ID
from step6.filter_below_driveable import filter_below_driveable


def test_only_free_below_lowest_driveable_surface_is_cleared():
    semantics = np.full((3, 3, 6), UNKNOWN_ID, dtype=np.uint8)
    mask = np.ones_like(semantics, dtype=bool)
    semantics[1, 1, 1] = OBSERVED_FREE_ID
    semantics[1, 1, 2] = OBSERVED_FREE_ID
    semantics[1, 1, 3] = 11
    semantics[1, 1, 4] = OBSERVED_FREE_ID
    semantics[0, 0, 1] = OBSERVED_FREE_ID
    updated, updated_mask, stats = filter_below_driveable(semantics, mask)
    assert updated[1, 1, 1] == UNKNOWN_ID
    assert updated[1, 1, 2] == UNKNOWN_ID
    assert not updated_mask[1, 1, 1]
    assert not updated_mask[1, 1, 2]
    assert updated[1, 1, 3] == 11
    assert updated[1, 1, 4] == OBSERVED_FREE_ID
    assert updated[0, 0, 1] == OBSERVED_FREE_ID
    assert stats["cleared_observed_free_voxels"] == 2
