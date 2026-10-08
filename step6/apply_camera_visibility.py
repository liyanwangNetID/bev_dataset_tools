"""Apply Step 6C camera visibility to formal Occ3D labels safely."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from project_paths import ALPASIM_DATA_ROOT, MANIFEST_ROOT, OUTCOME_ROOT, REPORT_ROOT
from step1.clip_manifest import CAMERA_NAMES
from step6.camera_visibility import (
    load_assignment_index,
    load_camera_contract,
    project_voxel_contract,
)
from step6.contract import GRID, OBSERVED_FREE_ID, UNKNOWN_ID
from step6.filter_below_driveable import filter_below_driveable

PRODUCTION_VISIBILITY_VERSION = "0.2.0"


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"row must be an object: {path}:{line_number}")
            yield value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_scene_contracts(
    scene_id: str,
    dataset_root: Path,
    assignments: dict[tuple[str, str], str],
    rectification_root: Path,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
    return {
        camera_name: load_camera_contract(
            scene_id,
            camera_name,
            dataset_root,
            assignments,
            rectification_root,
        )
        for camera_name in CAMERA_NAMES
    }


def compute_semantics_and_mask(
    semantics: np.ndarray,
    contracts: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, str]],
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, int]]]:
    original = semantics.copy()
    free = np.zeros(GRID.shape, dtype=bool)
    supervised = np.zeros(GRID.shape, dtype=bool)
    camera_stats = {}
    for camera_name in CAMERA_NAMES:
        extrinsic, k_rect, valid_mask, _ = contracts[camera_name]
        _, camera_free, camera_supervised, stats = project_voxel_contract(
            original, extrinsic, k_rect, valid_mask
        )
        free |= camera_free
        supervised |= camera_supervised
        camera_stats[camera_name] = stats
    updated = original.copy()
    updated[free & (updated == UNKNOWN_ID)] = OBSERVED_FREE_ID
    mask_camera = supervised & (updated != UNKNOWN_ID)
    return updated, mask_camera, camera_stats

def atomic_update_npz(
    label_path: Path,
    semantics: np.ndarray,
    mask_camera: np.ndarray,
    mask_lidar: np.ndarray,
) -> None:
    with tempfile.NamedTemporaryFile(
        dir=label_path.parent,
        prefix=label_path.name + ".",
        suffix=".tmp.npz",
        delete=False,
    ) as file:
        temporary = Path(file.name)
    try:
        np.savez_compressed(
            temporary,
            semantics=semantics,
            mask_camera=mask_camera,
            mask_lidar=mask_lidar,
        )
        with temporary.open("rb") as file:
            os.fsync(file.fileno())
        temporary.replace(label_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def apply_visibility(
    *,
    label_manifest: Path,
    outcome_root: Path,
    dataset_root: Path,
    assignments_path: Path,
    rectification_root: Path,
    summary_output: Path,
    audit_output: Path,
    limit: int | None,
    force: bool,
) -> dict[str, Any]:
    if (summary_output.exists() or audit_output.exists()) and not force:
        raise FileExistsError("Step 6C outputs exist; pass --force")

    assignments = load_assignment_index(assignments_path)
    active_scene = None
    active_contracts = None
    aggregate_camera = Counter()
    visible_class_counts = Counter()
    records = []

    for index, row in enumerate(read_jsonl(label_manifest), 1):
        if limit is not None and index > limit:
            break
        scene_id = str(row["scene_id"])
        if scene_id != active_scene:
            active_contracts = load_scene_contracts(
                scene_id,
                dataset_root,
                assignments,
                rectification_root,
            )
            active_scene = scene_id
        assert active_contracts is not None

        label_path = outcome_root / row["labels_path"]
        with np.load(label_path) as arrays:
            semantics = arrays["semantics"]
            existing_camera = arrays["mask_camera"]
            mask_lidar = arrays["mask_lidar"]

        if semantics.shape != GRID.shape or semantics.dtype != np.uint8:
            raise ValueError(f"invalid semantics contract: {label_path}")
        if mask_lidar.shape != GRID.shape or mask_lidar.dtype != np.bool_:
            raise ValueError(f"invalid mask_lidar contract: {label_path}")
        if mask_lidar.any():
            raise ValueError(f"mask_lidar must remain all false: {label_path}")

        semantics, mask_camera, camera_stats = compute_semantics_and_mask(semantics, active_contracts)
        semantics, mask_camera, below_road_stats = filter_below_driveable(
            semantics, mask_camera
        )
        if np.any(mask_camera & (semantics == UNKNOWN_ID)):
            raise RuntimeError(f"unknown voxel marked visible: {label_path}")

        if force or not np.array_equal(mask_camera, existing_camera):
            atomic_update_npz(
                label_path,
                semantics,
                mask_camera,
                mask_lidar,
            )

        values, counts = np.unique(
            semantics[mask_camera],
            return_counts=True,
        )
        visible_class_counts.update(
            {
                int(value): int(count)
                for value, count in zip(values, counts)
            }
        )
        aggregate_camera.update(
            {
                camera_name: stats["first_surface"]
                for camera_name, stats in camera_stats.items()
            }
        )
        known_count = int(np.count_nonzero(semantics != UNKNOWN_ID))
        free_count = int(np.count_nonzero(semantics == OBSERVED_FREE_ID))
        visible_free_count = int(np.count_nonzero(mask_camera & (semantics == OBSERVED_FREE_ID)))
        visible_count = int(mask_camera.sum())
        records.append(
            {
                "sample_id": row["sample_id"],
                "scene_id": scene_id,
                "labels_path": row["labels_path"],
                "known_voxels": known_count,
                "observed_free_voxels": free_count,
                "visible_observed_free_voxels": visible_free_count,
                "mask_camera_true": visible_count,
                "below_road_filter": below_road_stats,
                "visible_known_ratio": (
                    visible_count / known_count
                    if known_count
                    else 0.0
                ),
            }
        )

        if index == 1 or index % 500 == 0 or (limit is not None and index == limit):
            denominator = str(limit) if limit is not None else "all"
            print(
                f"[Step 6C Production] {index}/{denominator}: "
                f"{row['sample_id']} visible={visible_count}"
            )

    audit_text = "".join(
        json.dumps(record, separators=(",", ":")) + "\n"
        for record in records
    )
    ratios = [record["visible_known_ratio"] for record in records]
    visible_counts = [record["mask_camera_true"] for record in records]
    summary = {
        "visibility_version": PRODUCTION_VISIBILITY_VERSION,
        "record_count": len(records),
        "scene_count": len({record["scene_id"] for record in records}),
        "label_manifest": str(label_manifest),
        "label_manifest_sha256": sha256_file(label_manifest),
        "rectification_assignments": str(assignments_path),
        "rectification_assignments_sha256": sha256_file(assignments_path),
        "policy": "valid_rectified_camera_frustum unknown-to-17; mask_camera only through first known surface",
        "unknown_voxels_visible": False,
        "observed_free_generated": True,
        "observed_free_id": OBSERVED_FREE_ID,
        "observed_free_voxel_count": int(visible_class_counts.get(OBSERVED_FREE_ID, 0)) + int(sum(record["observed_free_voxels"] - record["visible_observed_free_voxels"] for record in records)),
        "visible_observed_free_voxel_count": int(visible_class_counts.get(OBSERVED_FREE_ID, 0)),
        "mask_lidar_policy": "all_false_no_lidar",
        "camera_contract_cache_policy": "active_scene_only",
        "mask_camera_true_minimum": min(visible_counts, default=0),
        "mask_camera_true_maximum": max(visible_counts, default=0),
        "mask_camera_true_mean": (
            sum(visible_counts) / len(visible_counts)
            if visible_counts
            else 0.0
        ),
        "visible_known_ratio_minimum": min(ratios, default=0.0),
        "visible_known_ratio_maximum": max(ratios, default=0.0),
        "visible_known_ratio_mean": (
            sum(ratios) / len(ratios)
            if ratios
            else 0.0
        ),
        "visible_class_voxel_counts": {
            str(key): value
            for key, value in sorted(visible_class_counts.items())
        },
        "aggregate_camera_zbuffer_visible": dict(aggregate_camera),
        "trial": limit is not None,
        "audit_sha256": hashlib.sha256(audit_text.encode()).hexdigest(),
    }

    audit_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(audit_text, encoding="utf-8")
    summary_output.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-manifest", type=Path, default=MANIFEST_ROOT / "occupancy_labels_v0.1.jsonl")
    parser.add_argument("--outcome-root", type=Path, default=OUTCOME_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=ALPASIM_DATA_ROOT)
    parser.add_argument("--assignments", type=Path, default=MANIFEST_ROOT / "rectification_assignments_v0.1.jsonl")
    parser.add_argument("--rectification-root", type=Path, default=OUTCOME_ROOT / "calibration" / "rectification_v0.1")
    parser.add_argument("--summary-output", type=Path, default=REPORT_ROOT / "occupancy_camera_visibility_summary_v0.1.json")
    parser.add_argument("--audit-output", type=Path, default=REPORT_ROOT / "occupancy_camera_visibility_audit_v0.1.jsonl")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    summary = apply_visibility(
        label_manifest=args.label_manifest,
        outcome_root=args.outcome_root,
        dataset_root=args.dataset_root,
        assignments_path=args.assignments,
        rectification_root=args.rectification_root,
        summary_output=args.summary_output,
        audit_output=args.audit_output,
        limit=args.limit,
        force=args.force,
    )
    print("Summary:", args.summary_output)
    print("Audit:", args.audit_output)
    print("Records:", summary["record_count"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
