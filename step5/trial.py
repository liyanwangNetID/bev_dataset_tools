#!/usr/bin/env python3
"""Generate a small four-camera F-theta rectification review trial."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from alpasim_driver.rectification import build_ftheta_rectifier_for_resolution
from alpasim_driver.schema import RectificationTargetConfig
from alpasim_grpc.v0 import sensorsim_pb2
from google.protobuf.json_format import ParseDict

from project_paths import ALPASIM_DATA_ROOT, MANIFEST_ROOT, OUTCOME_ROOT
from step1.clip_manifest import CAMERA_NAMES

TRIAL_VERSION = "0.1.0"
TARGET_WIDTH = 960
TARGET_HEIGHT = 540
WIDE_FOCAL = 480.0
TELE_FOCAL = 1925.175
PRINCIPAL_POINT = (479.5, 269.5)
DEFAULT_SCENES = (
    "test_clip_001",
    "test_clip_250",
    "test_clip_500",
    "test_clip_750",
    "test_clip_908",
)
RECTIFICATION_SOURCE = Path(
    "/home/lab/alpasim/src/driver/src/alpasim_driver/rectification.py"
)
SCHEMA_SOURCE = Path(
    "/home/lab/alpasim/src/driver/src/alpasim_driver/schema.py"
)


@dataclass(frozen=True, slots=True)
class RectificationAsset:
    camera_name: str
    focal_length: float
    map_x: np.ndarray
    map_y: np.ndarray
    valid_mask: np.ndarray
    valid_ratio: float
    rectifier: Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def target_config(camera_name: str) -> RectificationTargetConfig:
    focal = TELE_FOCAL if camera_name == "front_tele" else WIDE_FOCAL
    return RectificationTargetConfig(
        focal_length=(focal, focal),
        principal_point=PRINCIPAL_POINT,
        resolution_hw=(TARGET_HEIGHT, TARGET_WIDTH),
        radial=(),
        tangential=(),
        thin_prism=(),
        max_overscan_scale=1.0,
        safety_margin_px=0,
    )


def load_camera_proto(calibration_path: Path) -> Any:
    data = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    available = data.get("available_camera")
    if not isinstance(available, dict):
        raise ValueError(f"missing available_camera object: {calibration_path}")
    proto = sensorsim_pb2.AvailableCamerasReturn.AvailableCamera()
    ParseDict(available, proto)
    return proto


def build_asset(
    camera_name: str,
    calibration_path: Path,
    source_height: int,
    source_width: int,
) -> RectificationAsset:
    config = target_config(camera_name)
    proto = load_camera_proto(calibration_path)
    rectifier = build_ftheta_rectifier_for_resolution(
        camera_proto=proto,
        target_cfg=config,
        source_resolution_hw=(source_height, source_width),
    )
    map_x = np.asarray(rectifier._map_x)
    map_y = np.asarray(rectifier._map_y)
    if map_x.shape != (TARGET_HEIGHT, TARGET_WIDTH):
        raise ValueError(f"unexpected map shape for {camera_name}: {map_x.shape}")
    valid = (
        (map_x >= 0.0)
        & (map_x < source_width - 1.0)
        & (map_y >= 0.0)
        & (map_y < source_height - 1.0)
    )
    return RectificationAsset(
        camera_name=camera_name,
        focal_length=config.focal_length[0],
        map_x=map_x,
        map_y=map_y,
        valid_mask=valid,
        valid_ratio=float(np.mean(valid)),
        rectifier=rectifier,
    )


def read_manifest_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def select_trial_records(
    records: list[dict[str, Any]],
    scene_ids: tuple[str, ...],
) -> list[dict[str, Any]]:
    by_scene: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_scene.setdefault(record["scene_id"], []).append(record)
    selected = []
    for scene_id in scene_ids:
        candidates = by_scene.get(scene_id, [])
        if not candidates:
            raise ValueError(f"scene not present in manifest: {scene_id}")
        temporal = [item for item in candidates if not item["is_scene_start"]]
        source = temporal if temporal else candidates
        selected.append(source[len(source) // 2])
    return selected


def add_label(image: np.ndarray, text: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 38), (0, 0, 0), -1)
    cv2.putText(
        output,
        text,
        (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def fit_tile(image: np.ndarray, width: int = 640, height: int = 360) -> np.ndarray:
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def process_record(
    record: dict[str, Any],
    raw_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    scene_id = record["scene_id"]
    sample_id = record["sample_id"]
    sample_root = output_root / scene_id / sample_id
    raw_dir = sample_root / "raw"
    rectified_dir = sample_root / "rectified"
    mask_dir = sample_root / "valid_mask"
    map_dir = sample_root / "maps"
    for directory in (raw_dir, rectified_dir, mask_dir, map_dir):
        directory.mkdir(parents=True, exist_ok=True)

    top_tiles = []
    bottom_tiles = []
    camera_metadata = {}

    for camera_name in CAMERA_NAMES:
        camera = record["current_cameras"][camera_name]
        source_path = raw_root / camera["image_path"]
        source_bgr = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if source_bgr is None:
            raise ValueError(f"cannot decode image: {source_path}")
        source_height, source_width = source_bgr.shape[:2]
        calibration_path = raw_root / scene_id / "calibration" / f"{camera_name}.json"
        asset = build_asset(
            camera_name,
            calibration_path,
            source_height,
            source_width,
        )
        rectified_bgr = asset.rectifier.rectify(source_bgr)
        if rectified_bgr.shape[:2] != (TARGET_HEIGHT, TARGET_WIDTH):
            raise ValueError(
                f"unexpected rectified shape for {camera_name}: "
                f"{rectified_bgr.shape}"
            )
        mask_u8 = asset.valid_mask.astype(np.uint8) * 255
        masked_preview = rectified_bgr.copy()
        masked_preview[~asset.valid_mask] = (0, 0, 255)

        raw_output = raw_dir / f"{camera_name}.jpg"
        rectified_output = rectified_dir / f"{camera_name}.png"
        mask_output = mask_dir / f"{camera_name}.png"
        map_output = map_dir / f"{camera_name}.npz"
        shutil.copy2(source_path, raw_output)
        cv2.imwrite(str(rectified_output), rectified_bgr)
        cv2.imwrite(str(mask_output), mask_u8)
        np.savez_compressed(
            map_output,
            map_x=asset.map_x.astype(np.float32),
            map_y=asset.map_y.astype(np.float32),
            valid_mask=asset.valid_mask,
        )

        top_tiles.append(
            add_label(
                fit_tile(source_bgr),
                f"raw {camera_name} {source_width}x{source_height}",
            )
        )
        bottom_tiles.append(
            add_label(
                fit_tile(masked_preview),
                f"rectified {camera_name} valid={asset.valid_ratio:.3f}",
            )
        )
        camera_metadata[camera_name] = {
            "source_image": str(source_path.relative_to(raw_root)),
            "source_resolution": [source_width, source_height],
            "target_resolution": [TARGET_WIDTH, TARGET_HEIGHT],
            "focal_length": [asset.focal_length, asset.focal_length],
            "principal_point": list(PRINCIPAL_POINT),
            "K_rect": [
                [asset.focal_length, 0.0, PRINCIPAL_POINT[0]],
                [0.0, asset.focal_length, PRINCIPAL_POINT[1]],
                [0.0, 0.0, 1.0],
            ],
            "valid_pixel_ratio": asset.valid_ratio,
            "calibration_sha256": sha256_file(calibration_path),
            "raw_output": str(raw_output.relative_to(output_root)),
            "rectified_output": str(rectified_output.relative_to(output_root)),
            "valid_mask_output": str(mask_output.relative_to(output_root)),
            "map_output": str(map_output.relative_to(output_root)),
        }

    montage = np.vstack((np.hstack(top_tiles), np.hstack(bottom_tiles)))
    montage_path = sample_root / "review_montage.jpg"
    cv2.imwrite(str(montage_path), montage, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    metadata = {
        "trial_version": TRIAL_VERSION,
        "sample_id": sample_id,
        "scene_id": scene_id,
        "current_timestamp_ns": record["current_timestamp_ns"],
        "camera_order": list(CAMERA_NAMES),
        "rectification_backend": "alpasim_driver.rectification",
        "rectification_backend_sha256": sha256_file(RECTIFICATION_SOURCE),
        "schema_backend_sha256": sha256_file(SCHEMA_SOURCE),
        "target_distortion": "none",
        "cameras": camera_metadata,
        "review_montage": str(montage_path.relative_to(output_root)),
    }
    metadata_path = sample_root / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_ROOT / "occupancy_keyframes_v0.1.jsonl",
    )
    parser.add_argument("--raw-root", type=Path, default=ALPASIM_DATA_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTCOME_ROOT / "review" / "step5_rectification_trial_v0.1",
    )
    parser.add_argument("--scenes", nargs="*", default=list(DEFAULT_SCENES))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        if not args.force:
            raise FileExistsError(f"output exists: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    records = read_manifest_records(args.manifest)
    selected = select_trial_records(records, tuple(args.scenes))
    metadata = [
        process_record(record, args.raw_root.expanduser().resolve(), output_root)
        for record in selected
    ]
    valid_ratios = {
        camera: [item["cameras"][camera]["valid_pixel_ratio"] for item in metadata]
        for camera in CAMERA_NAMES
    }
    summary = {
        "trial_version": TRIAL_VERSION,
        "source_manifest": str(args.manifest),
        "source_manifest_sha256": sha256_file(args.manifest),
        "sample_count": len(metadata),
        "scene_ids": [item["scene_id"] for item in metadata],
        "rectification_backend_sha256": sha256_file(RECTIFICATION_SOURCE),
        "schema_backend_sha256": sha256_file(SCHEMA_SOURCE),
        "valid_pixel_ratio": {
            camera: {
                "minimum": min(values),
                "maximum": max(values),
                "mean": float(np.mean(values)),
            }
            for camera, values in valid_ratios.items()
        },
    }
    summary_path = output_root / "trial_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Trial output:", output_root)
    print("Summary:", summary_path)
    print("Samples:", len(metadata))
    for camera, stats in summary["valid_pixel_ratio"].items():
        print(camera, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
