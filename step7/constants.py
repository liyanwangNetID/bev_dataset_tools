from .config import Step7Paths
from pathlib import Path

CAMERA_ORDER = ("cross_left", "front_wide", "cross_right", "front_tele")
OCCUPANCY_SHAPE = (200, 200, 16)
UNKNOWN_LABEL = 255
FREE_LABEL = 17
_PATHS = Step7Paths.from_environment()
RAW_DATA_ROOT = _PATHS.raw_data_root
OUTCOME_ROOT = _PATHS.outcome_root
RECTIFICATION_ROOT = OUTCOME_ROOT / "calibration/rectification_v0.1"
KEYFRAME_MANIFEST = OUTCOME_ROOT / "manifests/occupancy_keyframes_v0.1.jsonl"
LABEL_MANIFEST = OUTCOME_ROOT / "manifests/occupancy_labels_v0.1.jsonl"
SOURCE_MANIFEST = OUTCOME_ROOT / "manifests/occupancy_sources_v0.1.jsonl"
