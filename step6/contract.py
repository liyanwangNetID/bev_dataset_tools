from __future__ import annotations
from dataclasses import dataclass

GENERATOR_VERSION = "0.2.0"
UNKNOWN_ID = 255
OBSERVED_FREE_ID = 17
OCC3D_CLASSES = {
  0:"others",1:"barrier",2:"bicycle",3:"bus",4:"car",
  5:"construction_vehicle",6:"motorcycle",7:"pedestrian",
  8:"traffic_cone",9:"trailer",10:"truck",11:"driveable_surface",
  12:"other_flat",13:"sidewalk",14:"terrain",15:"manmade",
  16:"vegetation",17:"observed_free_space",255:"ignore_unknown",
}
ACTOR_CLASS_TO_OCC3D = {
  "automobile":4,"person":7,"heavy_truck":10,"trailer":9,"bus":3,
  "rider":0,"protruding_object":0,"other_vehicle":0,"stroller":0,
  "train_or_tram_car":0,"animal":0,
}
@dataclass(frozen=True)
class VoxelGrid:
    x_min: float=-40.0; x_max: float=40.0
    y_min: float=-40.0; y_max: float=40.0
    z_min: float=-1.0; z_max: float=5.4
    voxel_size: float=0.4
    shape: tuple[int,int,int]=(200,200,16)
    axis_order: tuple[str,str,str]=("x","y","z")
    frame: str="current_ego"
GRID=VoxelGrid()
def map_actor_class(label_class: str) -> int:
    if label_class not in ACTOR_CLASS_TO_OCC3D:
        raise ValueError(f"unmapped actor label_class: {label_class!r}")
    return ACTOR_CLASS_TO_OCC3D[label_class]
def contract_dict() -> dict:
    return {
      "contract_version":"0.1","generator_version":GENERATOR_VERSION,
      "voxel_grid":{"extent_m":{"x":[-40.0,40.0],"y":[-40.0,40.0],"z":[-1.0,5.4]},"voxel_size_m":0.4,"shape":[200,200,16],"axis_order":["x","y","z"],"frame":"current_ego"},
      "occ3d_classes":{str(k):v for k,v in OCC3D_CLASSES.items()},
      "actor_class_to_occ3d":dict(ACTOR_CLASS_TO_OCC3D),
      "map_geometry":{"lane_ribbon":11},
      "unverifiable_space":255,"observed_free_requires_verified_ray":True,"traversability_policy":{"17":"traversable","0-16":"non_traversable","255":"non_traversable_unknown"},
      "source_policy":{"allowed":["local_clip_actor_current","local_clip_vector_map","frozen_keyframe_manifest","frozen_ego_pose"],"forbidden":["usdz","mesh","remote_download","online_simulator_service"]},
    }
