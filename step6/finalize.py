"""Freeze Step 6 semantic occupancy after all production audits pass."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from project_paths import MANIFEST_ROOT, OUTCOME_ROOT, REPORT_ROOT

FINALIZER_VERSION = "0.2.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def atomic_write(path: Path, text: str, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(f"output exists: {path}")
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as file:
        temporary = Path(file.name)
        file.write(text)
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def classify_edge_cases(diagnosis: dict[str, Any]) -> dict[str, Any]:
    """Validate diagnosis completeness without freezing dataset distributions."""
    records = diagnosis.get("records", [])
    if not isinstance(records, list):
        return {
            "status": "fail",
            "reason_counts": {},
            "zero_nonempty_count": 0,
            "full_visibility_count": 0,
            "violations": ["edge diagnosis records must be a list"],
        }

    zero_empty = [row for row in records if row.get("known_voxels") == 0]
    zero_nonempty = [
        row for row in records
        if row.get("known_voxels", 0) > 0 and row.get("mask_camera_true") == 0
    ]
    full = [
        row for row in records
        if row.get("known_voxels", 0) > 0
        and row.get("mask_camera_true") == row.get("known_voxels")
    ]
    violations = []
    classified_count = len(zero_empty) + len(zero_nonempty) + len(full)
    checks = (
        ("record_count", classified_count),
        ("zero_empty_count", len(zero_empty)),
        ("zero_nonempty_count", len(zero_nonempty)),
        ("full_visibility_count", len(full)),
    )
    for name, expected in checks:
        if diagnosis.get(name) != expected:
            violations.append(f"edge diagnosis {name} mismatch")
    for row in records:
        if row.get("recomputed_visible") != row.get("mask_camera_true"):
            violations.append(
                f"{row.get('sample_id', '<unknown>')}: recomputed visibility mismatch"
            )
    return {
        "status": "pass" if not violations else "fail",
        "policy": "diagnosed edge cases are recorded; dataset-scale distribution is not frozen",
        "reason_counts": {
            "empty_known_geometry": len(zero_empty),
            "diagnosed_zero_visibility_with_known_geometry": len(zero_nonempty),
            "diagnosed_full_visibility": len(full),
        },
        "zero_nonempty_count": len(zero_nonempty),
        "full_visibility_count": len(full),
        "violations": violations,
    }

def finalize(
    *,
    label_manifest: Path,
    labels_root: Path,
    label_summary_path: Path,
    visibility_summary_path: Path,
    final_audit_path: Path,
    edge_diagnosis_path: Path,
    output: Path,
    force: bool,
) -> dict[str, Any]:
    label_summary = read_json(label_summary_path)
    visibility_summary = read_json(visibility_summary_path)
    final_audit = read_json(final_audit_path)
    edge_diagnosis = read_json(edge_diagnosis_path)

    failures = []
    manifest_record_count = sum(
        1 for line in label_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    expected_record_count = final_audit.get("record_count")
    expected_scene_count = final_audit.get("scene_count")
    if not isinstance(expected_record_count, int) or expected_record_count <= 0:
        failures.append("step6 final audit record_count is invalid")
    if not isinstance(expected_scene_count, int) or expected_scene_count <= 0:
        failures.append("step6 final audit scene_count is invalid")
    if manifest_record_count != expected_record_count:
        failures.append(
            f"label manifest count is {manifest_record_count}, expected {expected_record_count}"
        )
    if final_audit.get("status") != "pass":
        failures.append("step6 final audit did not pass")
    if final_audit.get("failure_count") != 0:
        failures.append("step6 final audit has failures")
    if label_summary.get("record_count") != expected_record_count:
        failures.append(
            f"label summary record_count is {label_summary.get('record_count')}, expected {expected_record_count}"
        )
    if visibility_summary.get("record_count") != expected_record_count:
        failures.append(
            f"visibility summary record_count is {visibility_summary.get('record_count')}, expected {expected_record_count}"
        )
    if visibility_summary.get("unknown_voxels_visible") is not False:
        failures.append("unknown_voxels_visible must be false")
    if visibility_summary.get("observed_free_generated") is not False:
        failures.append("observed_free_generated must be false")
    if visibility_summary.get("mask_lidar_policy") != "all_false_no_lidar":
        failures.append("mask_lidar policy mismatch")

    label_count = sum(1 for _ in labels_root.rglob("labels.npz"))
    if label_count != expected_record_count:
        failures.append(
            f"label file count is {label_count}, expected {expected_record_count}"
        )

    edge_classification = classify_edge_cases(edge_diagnosis)
    if edge_classification["status"] != "pass":
        failures.extend(edge_classification["violations"])

    contract = {
        "step": 6,
        "contract_version": "0.1",
        "finalizer_version": FINALIZER_VERSION,
        "status": "frozen" if not failures else "failed",
        "record_count": expected_record_count,
        "scene_count": expected_scene_count,
        "label_file_count": label_count,
        "labels_root": str(labels_root),
        "label_manifest": str(label_manifest),
        "label_manifest_sha256": sha256_file(label_manifest),
        "label_summary": str(label_summary_path),
        "label_summary_sha256": sha256_file(label_summary_path),
        "visibility_summary": str(visibility_summary_path),
        "visibility_summary_sha256": sha256_file(visibility_summary_path),
        "final_audit": str(final_audit_path),
        "final_audit_sha256": sha256_file(final_audit_path),
        "edge_diagnosis": str(edge_diagnosis_path),
        "edge_diagnosis_sha256": sha256_file(edge_diagnosis_path),
        "grid": final_audit["frozen_contract"],
        "semantic_class_voxel_counts": final_audit["semantic_class_voxel_counts"],
        "visible_class_voxel_counts": final_audit["visible_class_voxel_counts"],
        "visibility_distribution": {
            "mask_camera_true": final_audit["mask_camera_true"],
            "visible_known_ratio": final_audit["visible_known_ratio"],
        },
        "edge_case_resolution": edge_classification,
        "source_boundary": {
            "input": f"{expected_scene_count} local Clips only",
            "usdz_used": False,
            "remote_download_used": False,
            "virtual_lidar_used": False,
        },
        "label_policy": {
            "semantic_ids": "Occ3D 0-17 plus 255",
            "recorded_actor_without_direct_occ3d_type": "others=0",
            "unverifiable_space": "unknown=255",
            "observed_free_17_generated": False,
            "driveable_surface_source": "VectorMap lane ribbons",
            "actor_source": "current Actor oriented 3D boxes",
            "mask_camera": "known semantic voxel centers, rectified valid mask, per-camera z-buffer",
            "mask_lidar": "all false",
        },
        "failures": failures,
    }
    atomic_write(output, json.dumps(contract, indent=2) + "\n", force)
    if failures:
        raise RuntimeError("Step 6 freeze failed: " + "; ".join(failures[:10]))
    return contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-manifest", type=Path, default=MANIFEST_ROOT / "occupancy_labels_v0.1.jsonl")
    parser.add_argument("--labels-root", type=Path, default=OUTCOME_ROOT / "occupancy" / "labels_v0.1")
    parser.add_argument("--label-summary", type=Path, default=REPORT_ROOT / "occupancy_label_summary_v0.1.json")
    parser.add_argument("--visibility-summary", type=Path, default=REPORT_ROOT / "occupancy_camera_visibility_summary_v0.1.json")
    parser.add_argument("--final-audit", type=Path, default=REPORT_ROOT / "step6_final_audit_v0.1.json")
    parser.add_argument("--edge-diagnosis", type=Path, default=REPORT_ROOT / "step6_visibility_edge_diagnosis_v0.1.json")
    parser.add_argument("--output", type=Path, default=REPORT_ROOT / "step6_dataset_contract_v0.1.json")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract = finalize(
        label_manifest=args.label_manifest,
        labels_root=args.labels_root,
        label_summary_path=args.label_summary,
        visibility_summary_path=args.visibility_summary,
        final_audit_path=args.final_audit,
        edge_diagnosis_path=args.edge_diagnosis,
        output=args.output,
        force=args.force,
    )
    print("Contract:", args.output)
    print("Status:", contract["status"])
    print("Labels:", contract["label_file_count"])
    print("Edge-case status:", contract["edge_case_resolution"]["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
