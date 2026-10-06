from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Step7Paths:
    raw_data_root: Path
    outcome_root: Path

    @classmethod
    def from_environment(cls):
        project_root = Path(os.environ.get(
            "BEV_DATASET_PROJECT_ROOT", "/home/lab/bev_alpasim_dataset_tools"
        )).expanduser()
        raw = Path(os.environ.get(
            "ALPASIM_RAW_DATA_ROOT", "/home/lab/data_from_alpasim"
        )).expanduser()
        outcome = Path(os.environ.get(
            "BEV_DATASET_OUTCOME_ROOT", str(project_root / "outcome")
        )).expanduser()
        return cls(raw.resolve(), outcome.resolve())

    @property
    def rectification_root(self):
        return self.outcome_root / "calibration/rectification_v0.1"

    @property
    def keyframe_manifest(self):
        return self.outcome_root / "manifests/occupancy_keyframes_v0.1.jsonl"

    @property
    def label_manifest(self):
        return self.outcome_root / "manifests/occupancy_labels_v0.1.jsonl"

    @property
    def source_manifest(self):
        return self.outcome_root / "manifests/occupancy_sources_v0.1.jsonl"
