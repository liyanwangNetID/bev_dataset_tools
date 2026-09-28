#!/usr/bin/env python3
"""Build the frozen Step 5 calibration census and rectification assets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from alpasim_driver.rectification import build_ftheta_rectifier_for_resolution
from alpasim_grpc.v0 import sensorsim_pb2
from google.protobuf.json_format import ParseDict

from project_paths import ALPASIM_DATA_ROOT, OUTCOME_ROOT, REPORT_ROOT
from step1.clip_manifest import CAMERA_NAMES, discover_clips
from step5.trial import (
    PRINCIPAL_POINT,
    RECTIFICATION_SOURCE,
    SCHEMA_SOURCE,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    target_config,
)

CONTRACT_VERSION = "0.1"
GENERATOR_VERSION = "0.1.0"
SOURCE_WIDTH = 854
SOURCE_HEIGHT = 480
LOW_VALID_RATIO = 0.90


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def split_calibration(data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    available = data.get("available_camera")
    if not isinstance(available, dict):
        raise ValueError("calibration has no available_camera object")
    intrinsics = available.get("intrinsics")
    extrinsics = available.get("rig_to_camera")
    if not isinstance(intrinsics, dict) or not isinstance(extrinsics, dict):
        raise ValueError("calibration intrinsics or extrinsics missing")
    intrinsic_contract = {
        "logical_id": available.get("logical_id"),
        "intrinsics": intrinsics,
    }
    extrinsic_contract = {
        "logical_id": available.get("logical_id"),
        "rig_to_camera": extrinsics,
    }
    return intrinsic_contract, extrinsic_contract


def parse_proto(data: dict[str, Any]) -> Any:
    available = data["available_camera"]
    proto = sensorsim_pb2.AvailableCamerasReturn.AvailableCamera()
    ParseDict(available, proto)
    return proto


def build_asset(camera_name: str, data: dict[str, Any]) -> dict[str, Any]:
    proto = parse_proto(data)
    config = target_config(camera_name)
    rectifier = build_ftheta_rectifier_for_resolution(
        camera_proto=proto,
        target_cfg=config,
        source_resolution_hw=(SOURCE_HEIGHT, SOURCE_WIDTH),
    )
    map_x = np.asarray(rectifier._map_x, dtype=np.float32)
    map_y = np.asarray(rectifier._map_y, dtype=np.float32)
    valid = (
        (map_x >= 0.0)
        & (map_x < SOURCE_WIDTH - 1.0)
        & (map_y >= 0.0)
        & (map_y < SOURCE_HEIGHT - 1.0)
    )
    focal = float(config.focal_length[0])
    return {
        "map_x": map_x,
        "map_y": map_y,
        "valid_mask": valid,
        "valid_pixel_ratio": float(valid.mean()),
        "K_rect": [
            [focal, 0.0, PRINCIPAL_POINT[0]],
            [0.0, focal, PRINCIPAL_POINT[1]],
            [0.0, 0.0, 1.0],
        ],
    }


def write_asset(
    asset_root: Path,
    asset_id: str,
    camera_name: str,
    intrinsic_hash: str,
    calibration_data: dict[str, Any],
    asset: dict[str, Any],
) -> dict[str, Any]:
    directory = asset_root / asset_id
    directory.mkdir(parents=True, exist_ok=True)
    maps_path = directory / "maps.npz"
    mask_path = directory / "valid_mask.png"
    metadata_path = directory / "metadata.json"
    np.savez_compressed(
        maps_path,
        map_x=asset["map_x"],
        map_y=asset["map_y"],
        valid_mask=asset["valid_mask"],
    )
    cv2.imwrite(str(mask_path), asset["valid_mask"].astype(np.uint8) * 255)
    metadata = {
        "contract_version": CONTRACT_VERSION,
        "generator_version": GENERATOR_VERSION,
        "asset_id": asset_id,
        "camera_name": camera_name,
        "intrinsic_hash": intrinsic_hash,
        "source_resolution": [SOURCE_WIDTH, SOURCE_HEIGHT],
        "target_resolution": [TARGET_WIDTH, TARGET_HEIGHT],
        "target_distortion": "none",
        "K_rect": asset["K_rect"],
        "valid_pixel_ratio": asset["valid_pixel_ratio"],
        "maps_sha256": sha256_file(maps_path),
        "valid_mask_sha256": sha256_file(mask_path),
        "source_intrinsics": split_calibration(calibration_data)[0],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * ratio)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=ALPASIM_DATA_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTCOME_ROOT / "calibration" / "rectification_v0.1",
    )
    parser.add_argument(
        "--census-output",
        type=Path,
        default=REPORT_ROOT / "step5_calibration_census_v0.1.jsonl",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=REPORT_ROOT / "step5_calibration_summary_v0.1.json",
    )
    parser.add_argument(
        "--low-valid-output",
        type=Path,
        default=REPORT_ROOT / "step5_low_valid_ratio_cases_v0.1.jsonl",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_root = args.raw_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        if not args.force:
            raise FileExistsError(f"output exists: {output_root}")
        shutil.rmtree(output_root)
    asset_root = output_root / "intrinsics_assets"
    asset_root.mkdir(parents=True)

    census_records = []
    unique_intrinsics: dict[str, tuple[str, dict[str, Any]]] = {}
    intrinsic_clip_counts = Counter()
    full_hash_counts = Counter()
    extrinsic_hash_counts = Counter()
    per_camera_intrinsic_hashes: dict[str, set[str]] = defaultdict(set)
    per_camera_full_hashes: dict[str, set[str]] = defaultdict(set)
    per_camera_extrinsic_hashes: dict[str, set[str]] = defaultdict(set)

    clips = discover_clips(raw_root)
    for index, clip in enumerate(clips, 1):
        for camera_name in CAMERA_NAMES:
            calibration_path = clip / "calibration" / f"{camera_name}.json"
            data = json.loads(calibration_path.read_text(encoding="utf-8"))
            intrinsics, extrinsics = split_calibration(data)
            intrinsic_hash = sha256_bytes(canonical_json(intrinsics).encode("utf-8"))
            extrinsic_hash = sha256_bytes(canonical_json(extrinsics).encode("utf-8"))
            full_hash = sha256_bytes(canonical_json(data).encode("utf-8"))
            asset_id = f"{camera_name}_{intrinsic_hash[:16]}"
            unique_intrinsics.setdefault(intrinsic_hash, (camera_name, data))
            intrinsic_clip_counts[intrinsic_hash] += 1
            extrinsic_hash_counts[extrinsic_hash] += 1
            full_hash_counts[full_hash] += 1
            per_camera_intrinsic_hashes[camera_name].add(intrinsic_hash)
            per_camera_extrinsic_hashes[camera_name].add(extrinsic_hash)
            per_camera_full_hashes[camera_name].add(full_hash)
            ftheta = data["available_camera"]["intrinsics"]["ftheta_param"]
            census_records.append(
                {
                    "contract_version": CONTRACT_VERSION,
                    "generator_version": GENERATOR_VERSION,
                    "clip_id": clip.name,
                    "camera_name": camera_name,
                    "calibration_path": str(calibration_path.relative_to(raw_root)),
                    "intrinsic_hash": intrinsic_hash,
                    "extrinsic_hash": extrinsic_hash,
                    "full_calibration_hash": full_hash,
                    "rectification_asset_id": asset_id,
                    "native_resolution": [
                        data["available_camera"]["intrinsics"]["resolution_w"],
                        data["available_camera"]["intrinsics"]["resolution_h"],
                    ],
                    "principal_point": [
                        ftheta["principal_point_x"],
                        ftheta["principal_point_y"],
                    ],
                    "max_angle": ftheta["max_angle"],
                }
            )
        if index == 1 or index % 50 == 0 or index == len(clips):
            print(f"Scanned {index}/{len(clips)}: {clip.name}")

    asset_metadata = {}
    for asset_index, (intrinsic_hash, (camera_name, data)) in enumerate(
        sorted(unique_intrinsics.items()),
        1,
    ):
        asset_id = f"{camera_name}_{intrinsic_hash[:16]}"
        asset = build_asset(camera_name, data)
        metadata = write_asset(
            asset_root,
            asset_id,
            camera_name,
            intrinsic_hash,
            data,
            asset,
        )
        metadata["clip_count"] = intrinsic_clip_counts[intrinsic_hash]
        metadata_path = asset_root / asset_id / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        asset_metadata[intrinsic_hash] = metadata
        if asset_index == 1 or asset_index % 100 == 0 or asset_index == len(unique_intrinsics):
            print(f"Built asset {asset_index}/{len(unique_intrinsics)}: {asset_id}")

    low_valid_records = []
    valid_ratios: dict[str, list[float]] = defaultdict(list)
    for record in census_records:
        metadata = asset_metadata[record["intrinsic_hash"]]
        ratio = metadata["valid_pixel_ratio"]
        record["valid_pixel_ratio"] = ratio
        valid_ratios[record["camera_name"]].append(ratio)
        if ratio < LOW_VALID_RATIO:
            low_valid_records.append(record)

    census_text = "".join(
        json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n"
        for record in census_records
    )
    low_valid_text = "".join(
        json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n"
        for record in low_valid_records
    )

    summary = {
        "contract_version": CONTRACT_VERSION,
        "generator_version": GENERATOR_VERSION,
        "clip_count": len(clips),
        "calibration_record_count": len(census_records),
        "unique_intrinsics_count": len(unique_intrinsics),
        "unique_intrinsics_by_camera": {
            camera: len(per_camera_intrinsic_hashes[camera])
            for camera in CAMERA_NAMES
        },
        "unique_extrinsics_by_camera": {
            camera: len(per_camera_extrinsic_hashes[camera])
            for camera in CAMERA_NAMES
        },
        "unique_full_calibrations_by_camera": {
            camera: len(per_camera_full_hashes[camera])
            for camera in CAMERA_NAMES
        },
        "source_resolution": [SOURCE_WIDTH, SOURCE_HEIGHT],
        "target_resolution": [TARGET_WIDTH, TARGET_HEIGHT],
        "rectification_backend": "alpasim_driver.rectification",
        "rectification_backend_sha256": sha256_file(RECTIFICATION_SOURCE),
        "schema_backend_sha256": sha256_file(SCHEMA_SOURCE),
        "low_valid_ratio_threshold": LOW_VALID_RATIO,
        "low_valid_record_count": len(low_valid_records),
        "valid_pixel_ratio_by_camera": {
            camera: {
                "minimum": min(values),
                "median": statistics.median(values),
                "mean": statistics.mean(values),
                "p95": percentile(values, 0.95),
                "maximum": max(values),
            }
            for camera, values in valid_ratios.items()
        },
        "census_sha256": sha256_bytes(census_text.encode("utf-8")),
    }
    contract = {
        "contract_version": CONTRACT_VERSION,
        "generator_version": GENERATOR_VERSION,
        "camera_order": list(CAMERA_NAMES),
        "source_resolution": [SOURCE_WIDTH, SOURCE_HEIGHT],
        "target_resolution": [TARGET_WIDTH, TARGET_HEIGHT],
        "target_distortion": "none",
        "wide_cross_K_rect": [
            [480.0, 0.0, PRINCIPAL_POINT[0]],
            [0.0, 480.0, PRINCIPAL_POINT[1]],
            [0.0, 0.0, 1.0],
        ],
        "tele_K_rect": [
            [1925.175, 0.0, PRINCIPAL_POINT[0]],
            [0.0, 1925.175, PRINCIPAL_POINT[1]],
            [0.0, 0.0, 1.0],
        ],
        "asset_key": "camera_name + canonical intrinsic hash",
        "valid_mask_rule": "0 <= map_x < source_width-1 and 0 <= map_y < source_height-1",
        "rectification_backend_sha256": summary["rectification_backend_sha256"],
        "schema_backend_sha256": summary["schema_backend_sha256"],
    }

    safe_write(args.census_output, census_text, args.force)
    safe_write(args.low_valid_output, low_valid_text, args.force)
    safe_write(args.summary_output, json.dumps(summary, indent=2) + "\n", args.force)
    (output_root / "contract.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")

    print("Output root:", output_root)
    print("Census:", args.census_output)
    print("Summary:", args.summary_output)
    print("Low-valid cases:", args.low_valid_output)
    print("Unique intrinsics:", summary["unique_intrinsics_count"])
    print("Low-valid records:", summary["low_valid_record_count"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
