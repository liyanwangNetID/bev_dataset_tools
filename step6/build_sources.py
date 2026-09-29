"""Step 6A integration with frozen keyframes and the unified Clip reader."""
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
from project_paths import ALPASIM_DATA_ROOT, MANIFEST_ROOT, REPORT_ROOT
from step2.clip_reader import DrivingClipReader
from step6.contract import GENERATOR_VERSION, contract_dict
from step6.source_geometry import normalize_sources

SOURCE_SCHEMA_VERSION = "0.1.1"
DEFAULT_ACTOR_TOLERANCE_NS = 100_000_000

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

def source_record(reader: DrivingClipReader, keyframe: dict[str, Any], actor_tolerance_ns: int) -> dict[str, Any]:
    timestamp_ns = int(keyframe["current_timestamp_ns"])
    actor_match = reader.get_actors_at(timestamp_ns, tolerance_ns=actor_tolerance_ns)
    if actor_match is None:
        raise RuntimeError(f"no Actor snapshot for {keyframe['sample_id']}")
    pose = np.asarray(keyframe["current_ego2global"], dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"invalid current_ego2global for {keyframe['sample_id']}")
    actors = actor_match.message.get("actors")
    if not isinstance(actors, list):
        raise ValueError(f"Actor message has no actors list: {keyframe['sample_id']}")
    vector_map = reader.get_vector_map()
    if vector_map.get("frame_id") != "map":
        raise ValueError(f"VectorMap frame is not map: {reader.clip_id}")
    sources = normalize_sources(actors=actors, vector_map=vector_map, ego2map=pose)
    actor_counts = Counter(item["occ3d_id"] for item in sources["actors"])
    source_label_counts = Counter(item["source_label_class"] for item in sources["actors"])
    return {
        "source_schema_version": SOURCE_SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "sample_id": keyframe["sample_id"],
        "scene_id": keyframe["scene_id"],
        "current_timestamp_ns": timestamp_ns,
        "actor_timestamp_ns": actor_match.stamp_ns,
        "actor_time_error_ns": actor_match.time_error_ns,
        "map_id": str(vector_map.get("map_id", "")),
        "map_revision": vector_map.get("revision"),
        "actor_count_source": len(actors),
        "actor_count_roi": len(sources["actors"]),
        "lane_count_source": len(vector_map.get("lanes", [])),
        "lane_ribbon_count_roi": len(sources["lane_ribbons"]),
        "actor_occ3d_counts": {str(k): v for k, v in sorted(actor_counts.items())},
        "actor_source_label_counts": dict(sorted(source_label_counts.items())),
        "sources": {
            "actors": sources["actors"],
            "lane_ribbons": [],
        },
        "map_source": {
            "kind": "clip_vector_map",
            "relative_path": f"{reader.clip_id}/map/vector_map.json",
            "frame_id": "map",
            "lane_ribbons_deferred_to_voxelization": True,
        },
    }

def build(*, keyframe_manifest: Path, dataset_root: Path, output: Path, summary_output: Path, contract_output: Path, force: bool, limit: int | None, actor_tolerance_ns: int) -> dict[str, Any]:
    records = []
    reader = None
    active_scene = None
    actor_time_errors = []
    actor_classes = Counter()
    source_labels = Counter()
    map_ids = set()
    for index, keyframe in enumerate(read_jsonl(keyframe_manifest), 1):
        if limit is not None and len(records) >= limit:
            break
        scene_id = str(keyframe["scene_id"])
        if scene_id != active_scene:
            reader = DrivingClipReader(dataset_root / scene_id)
            active_scene = scene_id
        assert reader is not None
        record = source_record(reader, keyframe, actor_tolerance_ns)
        records.append(record)
        actor_time_errors.append(record["actor_time_error_ns"])
        map_ids.add(record["map_id"])
        actor_classes.update({int(k): v for k, v in record["actor_occ3d_counts"].items()})
        source_labels.update(record["actor_source_label_counts"])
        if index == 1 or index % 500 == 0:
            print(f"[Step 6A] Built {index}: {record['sample_id']}")
    manifest_text = "".join(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n" for row in records)
    summary = {
        "source_schema_version": SOURCE_SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "source_keyframe_manifest": str(keyframe_manifest),
        "source_keyframe_manifest_sha256": sha256_file(keyframe_manifest),
        "record_count": len(records),
        "scene_count": len({row["scene_id"] for row in records}),
        "map_id_count": len(map_ids),
        "actor_timestamp_tolerance_ns": actor_tolerance_ns,
        "actor_time_error_ns_maximum": max(actor_time_errors, default=0),
        "actor_occ3d_counts": {str(k): v for k, v in sorted(actor_classes.items())},
        "actor_source_label_counts": dict(sorted(source_labels.items())),
        "actor_count_roi_total": sum(row["actor_count_roi"] for row in records),
        "lane_ribbon_count_roi_total": sum(row["lane_ribbon_count_roi"] for row in records),
        "lane_geometry_storage": "deferred_by_map_reference",
        "source_manifest_sha256": hashlib.sha256(manifest_text.encode()).hexdigest(),
        "limited_trial": limit is not None,
    }
    contract = contract_dict()
    contract.update({
        "source_schema_version": SOURCE_SCHEMA_VERSION,
        "source_keyframe_manifest": str(keyframe_manifest),
        "source_keyframe_manifest_sha256": summary["source_keyframe_manifest_sha256"],
        "actor_timestamp_tolerance_ns": actor_tolerance_ns,
    })
    safe_write(output, manifest_text, force)
    safe_write(summary_output, json.dumps(summary, indent=2, ensure_ascii=False) + "\n", force)
    safe_write(contract_output, json.dumps(contract, indent=2, ensure_ascii=False) + "\n", force)
    return summary

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyframe-manifest", type=Path, default=MANIFEST_ROOT / "occupancy_keyframes_v0.1.jsonl")
    parser.add_argument("--dataset-root", type=Path, default=ALPASIM_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=MANIFEST_ROOT / "occupancy_sources_v0.1.jsonl")
    parser.add_argument("--summary-output", type=Path, default=REPORT_ROOT / "occupancy_source_summary_v0.1.json")
    parser.add_argument("--contract-output", type=Path, default=REPORT_ROOT / "occupancy_source_contract_v0.1.json")
    parser.add_argument("--actor-tolerance-ns", type=int, default=DEFAULT_ACTOR_TOLERANCE_NS)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    summary = build(keyframe_manifest=args.keyframe_manifest, dataset_root=args.dataset_root, output=args.output, summary_output=args.summary_output, contract_output=args.contract_output, force=args.force, limit=args.limit, actor_tolerance_ns=args.actor_tolerance_ns)
    print("Sources:", args.output)
    print("Summary:", args.summary_output)
    print("Contract:", args.contract_output)
    print("Records:", summary["record_count"])
    print("Actor max time error ns:", summary["actor_time_error_ns_maximum"])
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
