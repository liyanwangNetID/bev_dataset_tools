#!/usr/bin/env python3
"""Build the frozen current/history keyframe manifest for occupancy training."""
from __future__ import annotations
import argparse, hashlib, json, os, statistics, tempfile
from collections import Counter
from pathlib import Path
from typing import Any
from project_paths import ALPASIM_DATA_ROOT, MANIFEST_ROOT, REPORT_ROOT
from step1.clip_manifest import CAMERA_NAMES, discover_clips
from step2.clip_reader import DrivingClipReader
from step3.profile import KEYFRAME_INTERVAL_NS, POSE_MAX_SPAN_NS, SYNC_TOLERANCE_NS, common_timestamps, nearest_timestamp, select_candidate_pairs

SCHEMA_VERSION = "0.1"
GENERATOR_VERSION = "0.1.1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_write(path: Path, text: str, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(f"output exists: {path}")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as file:
        temporary = Path(file.name)
        file.write(text)
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def matrix_lists(matrix: tuple[tuple[float, ...], ...]) -> list[list[float]]:
    return [list(row) for row in matrix]


def camera_records(reader: DrivingClipReader, stamp_ns: int, dataset_root: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for camera in CAMERA_NAMES:
        match = reader.camera_indexes[camera].exact(stamp_ns)
        if match is None:
            raise RuntimeError(f"missing exact {camera} frame at {stamp_ns}")
        frame = match.value
        result[camera] = {
            "frame_index": frame.frame_index,
            "timestamp_ns": frame.stamp_ns,
            "image_path": str(frame.image_path.relative_to(dataset_root)),
            "width": frame.width,
            "height": frame.height,
            "encoding": frame.encoding,
            "frame_id": frame.frame_id,
        }
    return result


def pose_or_rejection(reader: DrivingClipReader, stamp_ns: int, role: str) -> tuple[Any | None, dict[str, Any] | None]:
    pose = reader.get_ego_pose3d_at(stamp_ns, maximum_interpolation_span_ns=POSE_MAX_SPAN_NS)
    if pose is not None:
        return pose, None
    reader._ensure_ego_pose3d()
    index = reader._ego_pose3d_index
    assert index is not None
    before = index.at_or_before(stamp_ns)
    after = index.at_or_after(stamp_ns)
    if before is None or after is None:
        reason = f"{role}_pose_boundary"
        span = None
    else:
        span = after.timestamp_ns - before.timestamp_ns
        reason = f"{role}_pose_span" if span > POSE_MAX_SPAN_NS else f"{role}_pose_unavailable"
    return None, {
        "role": role,
        "timestamp_ns": stamp_ns,
        "reason": reason,
        "previous_timestamp_ns": before.timestamp_ns if before else None,
        "next_timestamp_ns": after.timestamp_ns if after else None,
        "interpolation_span_ns": span,
    }


def make_record(reader: DrivingClipReader, dataset_root: Path, history_ns: int, current_ns: int, scene_start: bool) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    history_pose, history_rejection = pose_or_rejection(reader, history_ns, "history")
    current_pose, current_rejection = pose_or_rejection(reader, current_ns, "current")
    rejections = [item for item in (history_rejection, current_rejection) if item is not None]
    if rejections:
        return None, rejections
    assert history_pose is not None and current_pose is not None
    calibrations = reader.get_all_camera_calibrations()
    return {
        "sample_schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "sample_id": f"{reader.clip_id}_{current_ns}",
        "scene_id": reader.clip_id,
        "is_scene_start": scene_start,
        "history_is_repeated_current": scene_start,
        "current_timestamp_ns": current_ns,
        "history_timestamp_ns": history_ns,
        "history_delta_ns": current_ns - history_ns,
        "camera_order": list(CAMERA_NAMES),
        "current_cameras": camera_records(reader, current_ns, dataset_root),
        "history_cameras": camera_records(reader, history_ns, dataset_root),
        "current_ego2global": matrix_lists(current_pose.ego2global),
        "history_ego2global": matrix_lists(history_pose.ego2global),
        "current_pose_exact": current_pose.exact,
        "history_pose_exact": history_pose.exact,
        "calibration_ids": {name: calibrations[name].logical_id for name in CAMERA_NAMES},
        "source_resolution": [854, 480],
        "rectification_status": "pending_step5",
    }, []


def inspect_clip(clip_dir: Path, dataset_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reader = DrivingClipReader(clip_dir)
    stamps = common_timestamps(reader)
    pairs = select_candidate_pairs(stamps)
    records = []
    rejections = []

    # A scene-start sample repeats the current observation as history. Choose
    # the earliest exact four-camera timestamp that also has a valid Ego pose.
    # The first common camera timestamp can precede ego_state coverage and must
    # not cause an otherwise valid scene to lose its required start sample.
    scene_start_ns = None
    for stamp_ns in stamps:
        pose = reader.get_ego_pose3d_at(
            stamp_ns,
            maximum_interpolation_span_ns=POSE_MAX_SPAN_NS,
        )
        if pose is not None:
            scene_start_ns = stamp_ns
            break

    if scene_start_ns is None:
        rejections.append(
            {
                "clip_id": reader.clip_id,
                "history_timestamp_ns": None,
                "current_timestamp_ns": None,
                "role": "scene_start",
                "timestamp_ns": None,
                "reason": "no_scene_start_pose",
                "previous_timestamp_ns": None,
                "next_timestamp_ns": None,
                "interpolation_span_ns": None,
            }
        )
    else:
        scene_start, rejected = make_record(
            reader,
            dataset_root,
            scene_start_ns,
            scene_start_ns,
            True,
        )
        if scene_start is None:
            raise RuntimeError(
                f"validated scene-start unexpectedly failed: {reader.clip_id}"
            )
        if rejected:
            raise RuntimeError(
                f"validated scene-start produced rejections: {reader.clip_id}"
            )
        records.append(scene_start)

    used_current = {record["current_timestamp_ns"] for record in records}
    for history_ns, current_ns in pairs:
        if current_ns in used_current:
            continue
        record, rejected = make_record(
            reader,
            dataset_root,
            history_ns,
            current_ns,
            False,
        )
        if record is not None:
            records.append(record)
            used_current.add(current_ns)
        for item in rejected:
            rejections.append(
                {
                    "clip_id": reader.clip_id,
                    "history_timestamp_ns": history_ns,
                    "current_timestamp_ns": current_ns,
                    **item,
                }
            )
    records.sort(key=lambda item: item["current_timestamp_ns"])
    return records, rejections

def validate_records(records: list[dict[str, Any]]) -> None:
    ids = [record["sample_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate sample_id")
    by_scene = {}
    for record in records:
        by_scene.setdefault(record["scene_id"], []).append(record)
        if record["camera_order"] != list(CAMERA_NAMES):
            raise RuntimeError("camera order mismatch")
        if record["is_scene_start"]:
            if record["history_delta_ns"] != 0 or not record["history_is_repeated_current"]:
                raise RuntimeError("invalid scene-start history")
        elif abs(record["history_delta_ns"] - KEYFRAME_INTERVAL_NS) > SYNC_TOLERANCE_NS:
            raise RuntimeError("invalid temporal history delta")
    for scene_records in by_scene.values():
        stamps = [item["current_timestamp_ns"] for item in scene_records]
        if stamps != sorted(stamps) or len(stamps) != len(set(stamps)):
            raise RuntimeError("scene timestamps are not unique and increasing")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ALPASIM_DATA_ROOT)
    parser.add_argument("--step1-manifest", type=Path, default=MANIFEST_ROOT / "clips_v0.1.jsonl")
    parser.add_argument("--step3-profile", type=Path, default=MANIFEST_ROOT / "occupancy_clip_profile_v0.1.jsonl")
    parser.add_argument("--output", type=Path, default=MANIFEST_ROOT / "occupancy_keyframes_v0.1.jsonl")
    parser.add_argument("--summary-output", type=Path, default=REPORT_ROOT / "occupancy_keyframe_summary_v0.1.json")
    parser.add_argument("--rejections-output", type=Path, default=REPORT_ROOT / "occupancy_temporal_rejections_v0.1.jsonl")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    step1_hash = sha256_file(args.step1_manifest)
    step3_hash = sha256_file(args.step3_profile)
    records = []
    rejections = []
    clips = discover_clips(dataset_root)
    for index, clip in enumerate(clips, 1):
        clip_records, clip_rejections = inspect_clip(clip, dataset_root)
        records.extend(clip_records)
        rejections.extend(clip_rejections)
        if index == 1 or index % 50 == 0 or index == len(clips):
            print(f"Built {index}/{len(clips)}: {clip.name}")
    validate_records(records)
    manifest_text = "".join(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n" for record in records)
    rejection_text = "".join(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n" for record in rejections)
    scene_counts = Counter(record["scene_id"] for record in records)
    reason_counts = Counter(record["reason"] for record in rejections)
    summary = {
        "sample_schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "source_step1_manifest_sha256": step1_hash,
        "source_step3_profile_sha256": step3_hash,
        "sample_count": len(records),
        "scene_count": len(scene_counts),
        "scene_start_sample_count": sum(record["is_scene_start"] for record in records),
        "temporal_sample_count": sum(not record["is_scene_start"] for record in records),
        "rejected_pair_count": len(rejections),
        "rejection_reason_counts": dict(reason_counts),
        "samples_per_scene": {"minimum": min(scene_counts.values()), "median": statistics.median(scene_counts.values()), "mean": statistics.mean(scene_counts.values()), "maximum": max(scene_counts.values())},
        "manifest_sha256": hashlib.sha256(manifest_text.encode()).hexdigest(),
    }
    safe_write(args.output, manifest_text, args.force)
    safe_write(args.rejections_output, rejection_text, args.force)
    safe_write(args.summary_output, json.dumps(summary, indent=2, ensure_ascii=False) + "\n", args.force)
    print("Manifest:", args.output)
    print("Summary:", args.summary_output)
    print("Rejections:", args.rejections_output)
    print("Samples:", summary["sample_count"])
    print("Scene starts:", summary["scene_start_sample_count"])
    print("Temporal samples:", summary["temporal_sample_count"])
    print("Rejected pairs:", summary["rejected_pair_count"])
    print("SHA-256:", summary["manifest_sha256"])
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
