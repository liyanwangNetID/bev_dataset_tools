from __future__ import annotations

import json
from pathlib import Path

import pytest

from step1.clip_manifest import CAMERA_NAMES, REQUIRED_RELATIVE_PATHS, build_summary, discover_clips, inspect_clip, read_timestamp_index


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def make_complete_clip(root: Path) -> Path:
    clip = root / "test_clip_001"
    clip.mkdir(parents=True)
    write_json(clip / "metadata.json", {"dataset_format_version": "0.2-batch", "status": "complete"})
    write_json(clip / "validation.json", {"valid": True, "checks": {"required": True}, "required_checks": ["required"], "first_sim_time_ns": 1_000_000_000, "last_sim_time_ns": 2_000_000_000, "sim_duration_sec": 1.0, "topic_counts": {}, "camera_statistics": {}})
    for relative in REQUIRED_RELATIVE_PATHS:
        path = clip / relative
        if path.exists():
            continue
        write_jsonl(path, [{"placeholder": True}]) if path.suffix == ".jsonl" else write_json(path, {})
    for camera in CAMERA_NAMES:
        camera_dir = clip / "cameras" / camera
        camera_dir.mkdir(parents=True)
        (camera_dir / "000000.jpg").write_bytes(b"jpeg")
        (camera_dir / "000001.jpg").write_bytes(b"jpeg")
        write_jsonl(camera_dir / "timestamps.jsonl", [{"stamp_ns": 1_000_000_000, "image_path": "000000.jpg"}, {"stamp_ns": 1_100_000_000, "image_path": "000001.jpg"}])
        write_json(clip / "calibration" / f"{camera}.json", {})
    return clip


def test_canonical_camera_order() -> None:
    assert CAMERA_NAMES == ("cross_left", "front_wide", "cross_right", "front_tele")


def test_discover_clips_numeric_order_and_exclusions(tmp_path: Path) -> None:
    for name in ("test_clip_010", "test_clip_002", "test_clip_001.tmp", "other"):
        (tmp_path / name).mkdir()
    assert [path.name for path in discover_clips(tmp_path)] == ["test_clip_002", "test_clip_010"]


def test_timestamp_index_quality_failures(tmp_path: Path) -> None:
    camera_dir = tmp_path / "camera"
    camera_dir.mkdir()
    (camera_dir / "000000.jpg").write_bytes(b"jpeg")
    (camera_dir / "orphan.jpg").write_bytes(b"jpeg")
    write_jsonl(camera_dir / "timestamps.jsonl", [{"stamp_ns": 20, "image_path": "000000.jpg"}, {"stamp_ns": 10, "image_path": "missing.jpg"}])
    summary, errors = read_timestamp_index(camera_dir / "timestamps.jsonl")
    assert errors == []
    assert summary["strictly_increasing"] is False
    assert summary["missing_image_count"] == 1
    assert summary["unindexed_jpeg_count"] == 1


def test_inspect_clip_is_complete_read_only_and_not_occupancy_gate(tmp_path: Path) -> None:
    clip = make_complete_clip(tmp_path)
    before = sorted((p.relative_to(clip), p.stat().st_size) for p in clip.rglob("*") if p.is_file())
    record = inspect_clip(clip, tmp_path)
    after = sorted((p.relative_to(clip), p.stat().st_size) for p in clip.rglob("*") if p.is_file())
    assert before == after
    assert record["manifest_usable"] is True
    assert record["missing_required_files"] == []
    assert list(record["cameras"]) == list(CAMERA_NAMES)
    assert "occupancy_eligible" not in record


def test_summary_determinism(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    record = inspect_clip(make_complete_clip(tmp_path), tmp_path)
    monkeypatch.setattr("step1.clip_manifest.utc_now_iso", lambda: "2026-09-28T00:00:00+00:00")
    assert build_summary([record]) == build_summary([record])
