from pathlib import Path

CAMERA_ORDER = ("cross_left", "front_wide", "cross_right", "front_tele")
OCCUPANCY_SHAPE = (200, 200, 16)
UNKNOWN_LABEL = 255
FREE_LABEL = 17
RAW_DATA_ROOT = Path("/home/lab/data_from_alpasim")
OUTCOME_ROOT = Path("/home/lab/bev_alpasim_dataset_tools/outcome")
RECTIFICATION_ROOT = OUTCOME_ROOT / "calibration/rectification_v0.1"
KEYFRAME_MANIFEST = OUTCOME_ROOT / "manifests/occupancy_keyframes_v0.1.jsonl"
LABEL_MANIFEST = OUTCOME_ROOT / "manifests/occupancy_labels_v0.1.jsonl"
SOURCE_MANIFEST = OUTCOME_ROOT / "manifests/occupancy_sources_v0.1.jsonl"
