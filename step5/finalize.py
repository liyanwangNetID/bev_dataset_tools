#!/usr/bin/env python3
"""Freeze Step 5 assignments and estimate deduplicated image materialization size."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from project_paths import ALPASIM_DATA_ROOT, MANIFEST_ROOT, OUTCOME_ROOT, REPORT_ROOT
from step1.clip_manifest import CAMERA_NAMES

CONTRACT_VERSION = "0.1"
GENERATOR_VERSION = "0.1.0"
TARGET_WIDTH = 960
TARGET_HEIGHT = 540


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def safe_write(path: Path, text: str, force: bool) -> None:
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


def build_assignments(
    census: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], str]]:
    assignments = []
    lookup = {}
    for record in census:
        key = (record["clip_id"], record["camera_name"])
        if key in lookup:
            raise RuntimeError(f"duplicate calibration assignment: {key}")
        lookup[key] = record["rectification_asset_id"]
        assignments.append(
            {
                "contract_version": CONTRACT_VERSION,
                "generator_version": GENERATOR_VERSION,
                "clip_id": record["clip_id"],
                "camera_name": record["camera_name"],
                "rectification_asset_id": record["rectification_asset_id"],
                "intrinsic_hash": record["intrinsic_hash"],
                "extrinsic_hash": record["extrinsic_hash"],
                "full_calibration_hash": record["full_calibration_hash"],
                "valid_pixel_ratio": record["valid_pixel_ratio"],
            }
        )
    assignments.sort(key=lambda item: (item["clip_id"], CAMERA_NAMES.index(item["camera_name"])))
    return assignments, lookup


def analyze_keyframes(
    records: list[dict[str, Any]],
    assignments: dict[tuple[str, str], str],
    raw_root: Path,
) -> dict[str, Any]:
    unique_images = set()
    missing_images = []
    asset_usage = Counter()
    role_references = Counter()
    scene_ids = set()

    for record in records:
        scene_id = record["scene_id"]
        scene_ids.add(scene_id)
        for role in ("current_cameras", "history_cameras"):
            for camera_name in CAMERA_NAMES:
                camera = record[role][camera_name]
                relative_path = camera["image_path"]
                unique_images.add(relative_path)
                role_references[(role, camera_name)] += 1
                key = (scene_id, camera_name)
                if key not in assignments:
                    raise RuntimeError(f"missing rectification assignment: {key}")
                asset_usage[assignments[key]] += 1

    source_bytes = 0
    for relative_path in sorted(unique_images):
        path = raw_root / relative_path
        if path.is_file():
            source_bytes += path.stat().st_size
        else:
            missing_images.append(relative_path)

    unique_count = len(unique_images)
    uncompressed_rgb_bytes = unique_count * TARGET_WIDTH * TARGET_HEIGHT * 3
    mask_bytes = unique_count * TARGET_WIDTH * TARGET_HEIGHT

    return {
        "scene_count": len(scene_ids),
        "sample_count": len(records),
        "unique_source_image_count": unique_count,
        "source_jpeg_bytes": source_bytes,
        "source_jpeg_gib": source_bytes / (1024 ** 3),
        "rectified_rgb_uncompressed_bytes": uncompressed_rgb_bytes,
        "rectified_rgb_uncompressed_gib": uncompressed_rgb_bytes / (1024 ** 3),
        "per_frame_valid_mask_uncompressed_bytes": mask_bytes,
        "per_frame_valid_mask_uncompressed_gib": mask_bytes / (1024 ** 3),
        "missing_image_count": len(missing_images),
        "first_missing_images": missing_images[:20],
        "asset_reference_counts": dict(sorted(asset_usage.items())),
        "role_reference_counts": {
            f"{role}:{camera}": count
            for (role, camera), count in sorted(role_references.items())
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--keyframes",
        type=Path,
        default=MANIFEST_ROOT / "occupancy_keyframes_v0.1.jsonl",
    )
    parser.add_argument(
        "--census",
        type=Path,
        default=REPORT_ROOT / "step5_calibration_census_v0.1.jsonl",
    )
    parser.add_argument(
        "--rectification-root",
        type=Path,
        default=OUTCOME_ROOT / "calibration" / "rectification_v0.1",
    )
    parser.add_argument("--raw-root", type=Path, default=ALPASIM_DATA_ROOT)
    parser.add_argument(
        "--assignments-output",
        type=Path,
        default=MANIFEST_ROOT / "rectification_assignments_v0.1.jsonl",
    )
    parser.add_argument(
        "--contract-output",
        type=Path,
        default=OUTCOME_ROOT / "calibration" / "step5_dataset_contract_v0.1.json",
    )
    parser.add_argument(
        "--storage-output",
        type=Path,
        default=REPORT_ROOT / "step5_storage_estimate_v0.1.json",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    keyframes = read_jsonl(args.keyframes)
    census = read_jsonl(args.census)
    assignments, lookup = build_assignments(census)
    analysis = analyze_keyframes(
        keyframes,
        lookup,
        args.raw_root.expanduser().resolve(),
    )
    if analysis["missing_image_count"]:
        raise RuntimeError(
            f"missing source images: {analysis['missing_image_count']}"
        )

    assignments_text = "".join(
        json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n"
        for record in assignments
    )
    contract_source = args.rectification_root / "contract.json"
    contract = {
        "contract_version": CONTRACT_VERSION,
        "generator_version": GENERATOR_VERSION,
        "camera_order": list(CAMERA_NAMES),
        "keyframe_manifest": str(args.keyframes),
        "keyframe_manifest_sha256": sha256_file(args.keyframes),
        "calibration_census": str(args.census),
        "calibration_census_sha256": sha256_file(args.census),
        "rectification_contract": str(contract_source),
        "rectification_contract_sha256": sha256_file(contract_source),
        "rectification_assignments": str(args.assignments_output),
        "rectification_assignments_sha256": hashlib.sha256(
            assignments_text.encode("utf-8")
        ).hexdigest(),
        "assignment_count": len(assignments),
        "rectification_asset_count": len(set(lookup.values())),
        "image_materialization_policy": {
            "deduplicate_by_raw_relative_path": True,
            "do_not_duplicate_current_history_images": True,
            "valid_masks_are_asset_level_not_frame_level": True,
            "default_mode": "on_demand_or_cached",
        },
    }

    safe_write(args.assignments_output, assignments_text, args.force)
    safe_write(
        args.contract_output,
        json.dumps(contract, indent=2) + "\n",
        args.force,
    )
    safe_write(
        args.storage_output,
        json.dumps(analysis, indent=2) + "\n",
        args.force,
    )

    print("Assignments:", args.assignments_output)
    print("Contract:", args.contract_output)
    print("Storage estimate:", args.storage_output)
    print("Assignments:", len(assignments))
    print("Assets:", contract["rectification_asset_count"])
    print("Unique source images:", analysis["unique_source_image_count"])
    print("Source JPEG GiB:", analysis["source_jpeg_gib"])
    print(
        "Uncompressed rectified RGB GiB:",
        analysis["rectified_rgb_uncompressed_gib"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
