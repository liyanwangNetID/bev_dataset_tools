"""Audit and freeze formal Step 6 semantic occupancy and camera visibility."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from project_paths import MANIFEST_ROOT, OUTCOME_ROOT, REPORT_ROOT
from step6.contract import GRID, OCC3D_CLASSES, UNKNOWN_ID

AUDIT_VERSION = "0.2.0"
ALLOWED_CLASS_IDS = frozenset(OCC3D_CLASSES)


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


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * ratio)
    return float(ordered[index])


def audit_labels(
    *,
    label_manifest: Path,
    outcome_root: Path,
    visibility_summary: Path,
    visibility_audit: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    label_rows = list(read_jsonl(label_manifest))
    visibility_rows = list(read_jsonl(visibility_audit))
    visibility_by_sample = {row["sample_id"]: row for row in visibility_rows}
    if len(visibility_by_sample) != len(visibility_rows):
        raise ValueError("duplicate sample_id in camera visibility audit")

    semantic_counts = Counter()
    visible_class_counts = Counter()
    scene_counts = Counter()
    mask_counts: list[int] = []
    visible_ratios: list[float] = []
    zero_visibility: list[dict[str, Any]] = []
    full_visibility: list[dict[str, Any]] = []
    failures: list[str] = []

    for index, row in enumerate(label_rows, 1):
        sample_id = row["sample_id"]
        scene_id = row["scene_id"]
        label_path = outcome_root / row["labels_path"]
        visibility = visibility_by_sample.get(sample_id)
        if visibility is None:
            failures.append(f"{sample_id}: missing visibility audit record")
            continue

        with np.load(label_path) as arrays:
            keys = set(arrays.files)
            if keys != {"semantics", "mask_camera", "mask_lidar"}:
                failures.append(f"{sample_id}: unexpected NPZ keys {sorted(keys)}")
                continue
            semantics = arrays["semantics"]
            mask_camera = arrays["mask_camera"]
            mask_lidar = arrays["mask_lidar"]

        if semantics.shape != GRID.shape or mask_camera.shape != GRID.shape or mask_lidar.shape != GRID.shape:
            failures.append(f"{sample_id}: bad array shape")
            continue
        if semantics.dtype != np.uint8 or mask_camera.dtype != np.bool_ or mask_lidar.dtype != np.bool_:
            failures.append(f"{sample_id}: bad array dtype")
            continue
        if mask_lidar.any():
            failures.append(f"{sample_id}: mask_lidar is not all false")
        if np.any(mask_camera & (semantics == UNKNOWN_ID)):
            failures.append(f"{sample_id}: unknown voxel marked camera-visible")

        values, counts = np.unique(semantics, return_counts=True)
        sample_classes = {int(value): int(count) for value, count in zip(values, counts)}
        unexpected = set(sample_classes) - ALLOWED_CLASS_IDS
        if unexpected:
            failures.append(f"{sample_id}: unexpected semantic IDs {sorted(unexpected)}")
        semantic_counts.update(sample_classes)

        visible_values, visible_counts = np.unique(semantics[mask_camera], return_counts=True)
        visible_class_counts.update(
            {int(value): int(count) for value, count in zip(visible_values, visible_counts)}
        )

        known_count = int(np.count_nonzero(semantics != UNKNOWN_ID))
        camera_count = int(mask_camera.sum())
        ratio = camera_count / known_count if known_count else 0.0
        mask_counts.append(camera_count)
        visible_ratios.append(ratio)
        scene_counts[scene_id] += 1

        if camera_count != int(visibility["mask_camera_true"]):
            failures.append(f"{sample_id}: NPZ/audit camera count mismatch")
        if known_count != int(visibility["known_voxels"]):
            failures.append(f"{sample_id}: NPZ/audit known count mismatch")

        case = {
            "sample_id": sample_id,
            "scene_id": scene_id,
            "labels_path": row["labels_path"],
            "known_voxels": known_count,
            "mask_camera_true": camera_count,
            "visible_known_ratio": ratio,
            "class_counts": {str(key): value for key, value in sorted(sample_classes.items())},
        }
        if camera_count == 0:
            zero_visibility.append(case)
        if known_count and camera_count == known_count:
            full_visibility.append(case)

        if index == 1 or index % 1000 == 0 or index == len(label_rows):
            print(f"[Step 6C Audit] {index}/{len(label_rows)}: {sample_id}")

    stored_visibility_summary = json.loads(visibility_summary.read_text(encoding="utf-8"))
    if stored_visibility_summary.get("record_count") != len(label_rows):
        failures.append("visibility summary record_count mismatch")
    if stored_visibility_summary.get("unknown_voxels_visible") is not False:
        failures.append("visibility summary does not freeze unknown_voxels_visible=false")
    if stored_visibility_summary.get("observed_free_generated") is not True:
        failures.append("visibility summary does not freeze observed_free_generated=true")
    if semantic_counts.get(17, 0) <= 0:
        failures.append("observed-free semantic ID 17 is absent")
    if visible_class_counts.get(17, 0) <= 0:
        failures.append("camera-visible observed-free semantic ID 17 is absent")

    summary = {
        "audit_version": AUDIT_VERSION,
        "status": "pass" if not failures else "fail",
        "record_count": len(label_rows),
        "scene_count": len(scene_counts),
        "label_manifest": str(label_manifest),
        "label_manifest_sha256": sha256_file(label_manifest),
        "visibility_summary": str(visibility_summary),
        "visibility_summary_sha256": sha256_file(visibility_summary),
        "visibility_audit": str(visibility_audit),
        "visibility_audit_sha256": sha256_file(visibility_audit),
        "semantic_class_voxel_counts": {str(key): value for key, value in sorted(semantic_counts.items())},
        "visible_class_voxel_counts": {str(key): value for key, value in sorted(visible_class_counts.items())},
        "mask_camera_true": {
            "minimum": min(mask_counts, default=0),
            "p01": percentile(mask_counts, 0.01),
            "median": percentile(mask_counts, 0.50),
            "p99": percentile(mask_counts, 0.99),
            "maximum": max(mask_counts, default=0),
            "mean": sum(mask_counts) / len(mask_counts) if mask_counts else 0.0,
        },
        "visible_known_ratio": {
            "minimum": min(visible_ratios, default=0.0),
            "p01": percentile(visible_ratios, 0.01),
            "median": percentile(visible_ratios, 0.50),
            "p99": percentile(visible_ratios, 0.99),
            "maximum": max(visible_ratios, default=0.0),
            "mean": sum(visible_ratios) / len(visible_ratios) if visible_ratios else 0.0,
        },
        "zero_visibility_sample_count": len(zero_visibility),
        "full_visibility_sample_count": len(full_visibility),
        "failure_count": len(failures),
        "failures": failures[:100],
        "frozen_contract": {
            "grid_shape": list(GRID.shape),
            "axis_order": ["x", "y", "z"],
            "semantics_dtype": "uint8",
            "mask_camera_dtype": "bool",
            "mask_lidar_dtype": "bool",
            "unknown_id": UNKNOWN_ID,
            "unknown_voxels_visible": False,
            "mask_lidar_policy": "all_false_no_lidar",
            "observed_free_generated": True,
            "camera_visibility_policy": "valid rectified frustum creates 17; mask_camera only through first known surface",
        },
    }
    edge_cases = zero_visibility + full_visibility
    return summary, edge_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-manifest", type=Path, default=MANIFEST_ROOT / "occupancy_labels_v0.1.jsonl")
    parser.add_argument("--outcome-root", type=Path, default=OUTCOME_ROOT)
    parser.add_argument("--visibility-summary", type=Path, default=REPORT_ROOT / "occupancy_camera_visibility_summary_v0.1.json")
    parser.add_argument("--visibility-audit", type=Path, default=REPORT_ROOT / "occupancy_camera_visibility_audit_v0.1.jsonl")
    parser.add_argument("--summary-output", type=Path, default=REPORT_ROOT / "step6_final_audit_v0.1.json")
    parser.add_argument("--edge-cases-output", type=Path, default=REPORT_ROOT / "step6_visibility_edge_cases_v0.1.jsonl")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (args.summary_output.exists() or args.edge_cases_output.exists()) and not args.force:
        raise FileExistsError("Step 6 audit outputs exist; pass --force")
    summary, edge_cases = audit_labels(
        label_manifest=args.label_manifest,
        outcome_root=args.outcome_root,
        visibility_summary=args.visibility_summary,
        visibility_audit=args.visibility_audit,
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.edge_cases_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.edge_cases_output.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in edge_cases),
        encoding="utf-8",
    )
    print("Summary:", args.summary_output)
    print("Edge cases:", args.edge_cases_output)
    print("Status:", summary["status"])
    print("Failures:", summary["failure_count"])
    print("Zero visibility:", summary["zero_visibility_sample_count"])
    print("Full visibility:", summary["full_visibility_sample_count"])
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
