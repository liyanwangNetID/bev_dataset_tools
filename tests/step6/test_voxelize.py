import numpy as np
from step6.contract import GRID, UNKNOWN_ID
from step6.source_geometry import normalize_actor, normalize_lane
from step6.voxelize import voxelize_actor, voxelize_lane_ribbon


def test_actor_overwrites_unknown():
    semantics = np.full(GRID.shape, UNKNOWN_ID, dtype=np.uint8)
    raw = {"track_id":"a","label_class":"automobile","is_static":False,"dimensions":{"x":2,"y":2,"z":2},"pose":{"position":{"x":0,"y":0,"z":1},"orientation":{"x":0,"y":0,"z":0,"w":1}}}
    actor = normalize_actor(raw, np.eye(4))
    assert actor is not None
    assert voxelize_actor(semantics, actor) > 0
    assert np.count_nonzero(semantics == 4) > 0


def test_actor_overrides_driveable_surface():
    semantics = np.full(GRID.shape, 11, dtype=np.uint8)
    raw = {"track_id":"a","label_class":"person","is_static":False,"dimensions":{"x":1,"y":1,"z":2},"pose":{"position":{"x":0,"y":0,"z":1},"orientation":{"x":0,"y":0,"z":0,"w":1}}}
    actor = normalize_actor(raw, np.eye(4))
    assert actor is not None
    voxelize_actor(semantics, actor)
    assert np.count_nonzero(semantics == 7) > 0


def test_lane_writes_single_surface_layer():
    semantics = np.full(GRID.shape, UNKNOWN_ID, dtype=np.uint8)
    raw = {"id":"L","left_boundary":{"points":[{"x":-2,"y":1,"z":0},{"x":2,"y":1,"z":0}]},"right_boundary":{"points":[{"x":-2,"y":-1,"z":0},{"x":2,"y":-1,"z":0}]}}
    lane = normalize_lane(raw, np.eye(4))
    assert lane is not None
    assert voxelize_lane_ribbon(semantics, lane) > 0
    occupied = np.argwhere(semantics == 11)
    assert len(np.unique(occupied[:, 2])) == 1

def test_production_defaults_are_not_trial_paths():
    from step6.voxelize import parse_args
    import sys
    original = sys.argv
    try:
        sys.argv = ["voxelize"]
        args = parse_args()
    finally:
        sys.argv = original
    assert args.limit is None
    assert "review" not in str(args.output_root)
    assert args.manifest_output.name == "occupancy_labels_v0.1.jsonl"

def test_voxelizer_uses_bounded_active_map_cache():
    from pathlib import Path
    source = Path(__file__).parents[2] / "step6" / "voxelize.py"
    text = source.read_text(encoding="utf-8")
    assert "map_cache: dict" not in text
    assert "active_map_path" in text
    assert "active_vector_map" in text
