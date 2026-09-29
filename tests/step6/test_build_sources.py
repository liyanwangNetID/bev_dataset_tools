import json
from pathlib import Path
import numpy as np
from step6.build_sources import source_record

class Match:
    stamp_ns=100
    time_error_ns=0
    message={"actors":[{"track_id":"a","label_class":"person","is_static":False,"dimensions":{"x":1,"y":1,"z":2},"pose":{"position":{"x":2,"y":0,"z":1},"orientation":{"x":0,"y":0,"z":0,"w":1}}}]}
class Reader:
    clip_id="clip"
    def get_actors_at(self, timestamp_ns, tolerance_ns): return Match()
    def get_vector_map(self):
        return {"frame_id":"map","map_id":"m","revision":1,"lanes":[{"id":"L","left_boundary":{"points":[{"x":0,"y":1,"z":0},{"x":3,"y":1,"z":0}]},"right_boundary":{"points":[{"x":0,"y":-1,"z":0},{"x":3,"y":-1,"z":0}]}}]}
def test_source_record_uses_frozen_fields():
    row=source_record(Reader(),{"sample_id":"s","scene_id":"clip","current_timestamp_ns":100,"current_ego2global":np.eye(4).tolist()},100)
    assert row["actor_time_error_ns"]==0
    assert row["actor_occ3d_counts"]=={"7":1}
    assert row["lane_ribbon_count_roi"]==1
    assert row["sources"]["lane_ribbons"] == []
    assert row["map_source"]["lane_ribbons_deferred_to_voxelization"] is True
