import numpy as np
from step6.contract import GRID
from step6.diagnose_visibility_edges import metric_bounds, quadrants


def test_edge_geometry_helpers():
    mask=np.zeros(GRID.shape,dtype=bool)
    mask[100,100,2]=True
    bounds=metric_bounds(mask)
    assert bounds is not None
    assert bounds["x"][0] == bounds["x"][1]
    assert quadrants(mask)["front_left"] == 1
