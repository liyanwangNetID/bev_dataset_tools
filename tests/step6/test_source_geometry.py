import numpy as np
from step6.source_geometry import normalize_actor,normalize_lane
def actor(x=5): return {"track_id":"7","label_class":"automobile","is_static":False,"dimensions":{"x":4,"y":2,"z":2},"pose":{"position":{"x":x,"y":0,"z":1},"orientation":{"x":0,"y":0,"z":0,"w":1}}}
def test_actor():
    out=normalize_actor(actor(),np.eye(4)); assert out["occ3d_id"]==4 and len(out["corners_ego_m"])==8
def test_actor_roi(): assert normalize_actor(actor(100),np.eye(4)) is None
def test_lane():
    lane={"id":"L","left_boundary":{"points":[{"x":0,"y":1,"z":0},{"x":2,"y":1,"z":0}]},"right_boundary":{"points":[{"x":0,"y":-1,"z":0},{"x":2,"y":-1,"z":0}]}}
    out=normalize_lane(lane,np.eye(4)); assert out["occ3d_id"]==11 and out["triangle_indices"]==[[0,1,2],[1,3,2]]
