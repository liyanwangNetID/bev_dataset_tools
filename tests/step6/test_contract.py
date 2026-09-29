import pytest
from step6.contract import ACTOR_CLASS_TO_OCC3D,GRID,OCC3D_CLASSES,map_actor_class
def test_grid(): assert GRID.shape==(200,200,16) and GRID.voxel_size==0.4 and GRID.axis_order==("x","y","z")
def test_ids(): assert set(OCC3D_CLASSES)==set(range(18))|{255}
def test_mapping_complete():
    expected={"automobile","person","heavy_truck","trailer","rider","bus","protruding_object","other_vehicle","stroller","train_or_tram_car","animal"}
    assert set(ACTOR_CLASS_TO_OCC3D)==expected and map_actor_class("automobile")==4 and map_actor_class("rider")==0
    with pytest.raises(ValueError): map_actor_class("schema_drift")
