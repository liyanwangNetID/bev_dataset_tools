#!/usr/bin/env python3
"""Profile Raw Clips for camera-only temporal occupancy training."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import statistics
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from project_paths import ALPASIM_DATA_ROOT, MANIFEST_ROOT, REPORT_ROOT
from step1.clip_manifest import CAMERA_NAMES, discover_clips
from step2.clip_reader import DrivingClipReader
from step2.temporal_index import synchronize_cameras_at

PROFILE_VERSION = "0.1"
SCRIPT_VERSION = "0.1.1"
SYNC_TOLERANCE_NS = 5_000_000
KEYFRAME_INTERVAL_NS = 500_000_000
POSE_MAX_SPAN_NS = 200_000_000


def percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * ratio)]


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


def common_timestamps(reader: DrivingClipReader) -> list[int]:
    return sorted(set.intersection(*(set(reader.camera_indexes[name].timestamps_ns) for name in CAMERA_NAMES)))


def build_grid(start_ns: int, end_ns: int) -> list[int]:
    if end_ns < start_ns:
        return []
    return list(range(start_ns, end_ns + 1, KEYFRAME_INTERVAL_NS))


def nearest_timestamp(stamps: list[int], target_ns: int, tolerance_ns: int) -> int | None:
    from bisect import bisect_left
    position = bisect_left(stamps, target_ns)
    candidates = []
    if position > 0:
        candidates.append(stamps[position - 1])
    if position < len(stamps):
        candidates.append(stamps[position])
    if not candidates:
        return None
    selected = min(candidates, key=lambda value: (abs(value - target_ns), value))
    return selected if abs(selected - target_ns) <= tolerance_ns else None


def select_candidate_pairs(stamps: list[int]) -> list[tuple[int, int]]:
    """Select non-overdense 2 Hz pairs from actual four-camera common stamps."""
    candidates = []
    for current_ns in stamps:
        history_ns = nearest_timestamp(stamps, current_ns - KEYFRAME_INTERVAL_NS, SYNC_TOLERANCE_NS)
        if history_ns is not None and history_ns < current_ns:
            candidates.append((history_ns, current_ns))
    selected = []
    last_current = None
    for history_ns, current_ns in candidates:
        if last_current is None or current_ns - last_current >= KEYFRAME_INTERVAL_NS:
            selected.append((history_ns, current_ns))
            last_current = current_ns
    return selected


def pair_poses_available(reader: DrivingClipReader, history_ns: int, current_ns: int) -> tuple[bool, int, int]:
    exact = 0
    interpolated = 0
    for stamp_ns in (history_ns, current_ns):
        pose = reader.get_ego_pose3d_at(stamp_ns, maximum_interpolation_span_ns=POSE_MAX_SPAN_NS)
        if pose is None:
            return False, exact, interpolated
        if pose.exact:
            exact += 1
        else:
            interpolated += 1
    return True, exact, interpolated


def inspect_clip(clip_dir: Path) -> dict[str, Any]:
    reader = DrivingClipReader(clip_dir)
    common = common_timestamps(reader)
    first = max(index.first_timestamp_ns for index in reader.camera_indexes.values() if index.first_timestamp_ns is not None)
    last = min(index.last_timestamp_ns for index in reader.camera_indexes.values() if index.last_timestamp_ns is not None)
    grid = build_grid(first, last)
    synchronized = {}
    skews = []
    errors = []
    for target_ns in grid:
        group = synchronize_cameras_at(reader.camera_indexes, target_ns, tolerance_ns=SYNC_TOLERANCE_NS, prefer_exact=True)
        if group is not None:
            synchronized[target_ns] = group
            skews.append(group.maximum_skew_ns)
            errors.append(group.maximum_target_error_ns)

    grid_pair_count = sum(target_ns - KEYFRAME_INTERVAL_NS in synchronized for target_ns in synchronized)
    selected_pairs = select_candidate_pairs(common)
    valid_pairs = 0
    pose_exact = 0
    pose_interpolated = 0
    for history_ns, current_ns in selected_pairs:
        available, exact, interpolated = pair_poses_available(reader, history_ns, current_ns)
        if available:
            valid_pairs += 1
            pose_exact += exact
            pose_interpolated += interpolated

    diagnostics = reader.get_camera_calibration_diagnostics()
    calibration_complete = len(diagnostics) == len(CAMERA_NAMES)
    resolution_pairs = sorted({f"{item.recorded_width}x{item.recorded_height}->{item.calibration_width}x{item.calibration_height}" for item in diagnostics})
    reasons = []
    if not common:
        reasons.append("no_exact_four_camera_timestamp")
    if valid_pairs == 0:
        reasons.append("no_valid_candidate_temporal_pair")
    if not calibration_complete:
        reasons.append("incomplete_calibration")
    if any(item.anisotropy > 0.001 for item in diagnostics):
        reasons.append("large_resolution_anisotropy")

    return {
        "profile_version": PROFILE_VERSION,
        "profile_builder_version": SCRIPT_VERSION,
        "clip_id": reader.clip_id,
        "duration_ns": reader.duration_ns,
        "camera_frame_counts": {name: len(reader.camera_indexes[name]) for name in CAMERA_NAMES},
        "exact_common_timestamp_count": len(common),
        "grid_target_count": len(grid),
        "synchronized_grid_count": len(synchronized),
        "synchronized_grid_ratio": len(synchronized) / len(grid) if grid else 0.0,
        "grid_temporal_pair_count": grid_pair_count,
        "candidate_temporal_pair_count": len(selected_pairs),
        "valid_temporal_pair_count": valid_pairs,
        "maximum_camera_skew_ns": max(skews) if skews else None,
        "maximum_target_error_ns": max(errors) if errors else None,
        "pose_exact_query_count": pose_exact,
        "pose_interpolated_query_count": pose_interpolated,
        "calibration_complete": calibration_complete,
        "resolution_pairs": resolution_pairs,
        "resolution_matches_calibration": all(item.resolution_matches for item in diagnostics),
        "maximum_scale_anisotropy": max((item.anisotropy for item in diagnostics), default=None),
        "temporal_eligible": valid_pairs > 0 and calibration_complete,
        "ineligibility_reasons": reasons,
    }


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = [record.get("valid_temporal_pair_count", 0) for record in records]
    grid_pairs = [record.get("grid_temporal_pair_count", 0) for record in records]
    ratios = [record.get("synchronized_grid_ratio", 0.0) for record in records]
    reason_counts = Counter(reason for record in records for reason in record.get("ineligibility_reasons", []))
    resolution_counts = Counter(pair for record in records for pair in record.get("resolution_pairs", []))
    return {
        "profile_version": PROFILE_VERSION,
        "profile_builder_version": SCRIPT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_clips": len(records),
        "temporal_eligible_clips": sum(bool(record.get("temporal_eligible")) for record in records),
        "total_grid_temporal_pairs": sum(grid_pairs),
        "total_valid_temporal_pairs": sum(pairs),
        "valid_temporal_pairs_per_clip": {
            "minimum": min(pairs) if pairs else None,
            "median": statistics.median(pairs) if pairs else None,
            "mean": statistics.mean(pairs) if pairs else None,
            "p95": percentile(pairs, 0.95),
            "maximum": max(pairs) if pairs else None,
        },
        "synchronized_grid_ratio": {
            "minimum": min(ratios) if ratios else None,
            "median": statistics.median(ratios) if ratios else None,
            "mean": statistics.mean(ratios) if ratios else None,
            "p95": percentile(ratios, 0.95),
            "maximum": max(ratios) if ratios else None,
        },
        "ineligibility_reason_counts": dict(reason_counts),
        "resolution_pair_counts": dict(resolution_counts),
        "clips_with_native_resolution_match": sum(bool(record.get("resolution_matches_calibration")) for record in records),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ALPASIM_DATA_ROOT)
    parser.add_argument("--profile-output", type=Path, default=MANIFEST_ROOT / "occupancy_clip_profile_v0.1.jsonl")
    parser.add_argument("--summary-output", type=Path, default=REPORT_ROOT / "occupancy_feasibility_v0.1.json")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    clips = discover_clips(args.dataset_root.expanduser().resolve())
    records = []
    for index, clip in enumerate(clips, 1):
        try:
            records.append(inspect_clip(clip))
        except Exception as exc:
            records.append({
                "profile_version": PROFILE_VERSION,
                "profile_builder_version": SCRIPT_VERSION,
                "clip_id": clip.name,
                "temporal_eligible": False,
                "grid_temporal_pair_count": 0,
                "candidate_temporal_pair_count": 0,
                "valid_temporal_pair_count": 0,
                "synchronized_grid_ratio": 0.0,
                "resolution_pairs": [],
                "resolution_matches_calibration": False,
                "ineligibility_reasons": [f"profile_error:{type(exc).__name__}:{exc}"],
            })
        if index == 1 or index % 50 == 0 or index == len(clips):
            print(f"Profiled {index}/{len(clips)}: {clip.name}")
    text = "".join(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n" for record in records)
    summary = build_summary(records)
    safe_write(args.profile_output, text, args.force)
    safe_write(args.summary_output, json.dumps(summary, indent=2, ensure_ascii=False) + "\n", args.force)
    print("Profile:", args.profile_output)
    print("Summary:", args.summary_output)
    print("SHA-256:", hashlib.sha256(text.encode()).hexdigest())
    print("Eligible clips:", summary["temporal_eligible_clips"])
    print("Grid pairs:", summary["total_grid_temporal_pairs"])
    print("Candidate valid pairs:", summary["total_valid_temporal_pairs"])
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
