import numpy as np
from step6.camera_visibility_review import quadrant_counts
from step6.contract import GRID

def test_quadrant_counts_follow_ego_axes():
    mask = np.zeros(GRID.shape, dtype=bool)
    mask[150,150,0] = True
    mask[150,50,0] = True
    mask[50,150,0] = True
    mask[50,50,0] = True
    assert quadrant_counts(mask) == {"front_left":1,"front_right":1,"rear_left":1,"rear_right":1}
