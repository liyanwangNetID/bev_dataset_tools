from __future__ import annotations
import math
import numpy as np
from .contract import GRID, map_actor_class

def quaternion_matrix(q: dict) -> np.ndarray:
    x,y,z,w=(float(q[k]) for k in ("x","y","z","w")); n=math.sqrt(x*x+y*y+z*z+w*w)
    if not math.isfinite(n) or n<=0: raise ValueError("invalid quaternion")
    x,y,z,w=x/n,y/n,z/n,w/n
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def pose_matrix(pose: dict) -> np.ndarray:
    m=np.eye(4); m[:3,:3]=quaternion_matrix(pose["orientation"]); m[:3,3]=[float(pose["position"][k]) for k in ("x","y","z")]; return m
def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    p=np.asarray(points,float); h=np.c_[p,np.ones(len(p))]; return (h@np.asarray(matrix,float).T)[:,:3]
def intersects_grid(points: np.ndarray) -> bool:
    lo=points.min(0); hi=points.max(0)
    return bool(hi[0]>=GRID.x_min and lo[0]<GRID.x_max and hi[1]>=GRID.y_min and lo[1]<GRID.y_max and hi[2]>=GRID.z_min and lo[2]<GRID.z_max)
def normalize_actor(actor: dict, ego2map: np.ndarray) -> dict|None:
    dims=np.array([float(actor["dimensions"][k]) for k in ("x","y","z")])
    if not np.isfinite(dims).all() or np.any(dims<=0): raise ValueError("invalid actor dimensions")
    actor2ego=np.linalg.inv(np.asarray(ego2map,float))@pose_matrix(actor["pose"])
    signs=np.array([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],[1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]],float)
    corners=transform_points(actor2ego,signs*dims/2)
    if not intersects_grid(corners): return None
    label=str(actor["label_class"])
    return {"track_id":str(actor["track_id"]),"source_label_class":label,"occ3d_id":map_actor_class(label),"is_static":bool(actor["is_static"]),"dimensions_xyz_m":dims.tolist(),"actor2ego":actor2ego.tolist(),"corners_ego_m":corners.tolist()}
def normalize_lane(lane: dict, ego2map: np.ndarray) -> dict|None:
    def pts(name): return np.asarray([[float(p[k]) for k in ("x","y","z")] for p in lane[name].get("points",[])])
    left,right=pts("left_boundary"),pts("right_boundary")
    if len(left)<2 or len(right)<2:return None
    n=min(len(left),len(right)); map2ego=np.linalg.inv(np.asarray(ego2map,float)); left=transform_points(map2ego,left[:n]); right=transform_points(map2ego,right[:n])
    vertices=np.empty((2*n,3)); vertices[0::2]=left; vertices[1::2]=right
    if not intersects_grid(vertices): return None
    faces=[]
    for i in range(n-1): a,b,c,d=2*i,2*i+1,2*i+2,2*i+3; faces.extend([[a,b,c],[b,d,c]])
    return {"lane_id":str(lane["id"]),"occ3d_id":11,"vertices_ego_m":vertices.tolist(),"triangle_indices":faces}
def normalize_sources(actors, vector_map:dict, ego2map:np.ndarray)->dict:
    aa=[x for a in actors if (x:=normalize_actor(a,ego2map)) is not None]
    ll=[x for l in vector_map.get("lanes",[]) if (x:=normalize_lane(l,ego2map)) is not None]
    return {"actors":aa,"lane_ribbons":ll}
