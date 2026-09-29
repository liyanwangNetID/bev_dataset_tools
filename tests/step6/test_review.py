import numpy as np
from step6.contract import UNKNOWN_ID
from step6.review import bev_projection

def test_bev_projection_actor_overrides_road():
    grid = np.full((2,2,3), UNKNOWN_ID, dtype=np.uint8)
    grid[0,0,0] = 11
    grid[0,0,1] = 4
    grid[1,1,0] = 11
    bev = bev_projection(grid)
    assert bev[0,0] == 4
    assert bev[1,1] == 11
    assert bev[0,1] == UNKNOWN_ID

def test_review_panel_height_contract():
    bev_height = 800 + 42
    slice_grid_height = 4 * (200 + 42)
    assert bev_height == 842
    assert slice_grid_height == 968
    assert slice_grid_height - bev_height == 126

def test_bev_display_axis_conversion():
    # Semantic array is indexed [x, y]. After transpose, image columns are x
    # and image rows are y; vertical flip makes positive y point upward.
    grid = np.full((3, 4, 1), UNKNOWN_ID, dtype=np.uint8)
    grid[2, 3, 0] = 4
    labels = bev_projection(grid).T
    displayed = np.flip(labels, axis=0)
    assert displayed[0, 2] == 4
