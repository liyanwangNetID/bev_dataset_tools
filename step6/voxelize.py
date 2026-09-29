"""Step 6B Occ3D semantic voxelization from compact Step 6A sources."""
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
from step6.contract import GRID, UNKNOWN_ID
from step6.source_geometry import normalize_lane

VOXELIZER_VERSION = "0.1.2"
LABEL_SCHEMA_VERSION = "0.1"


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
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=path.name + ".", suffix=".tmp", delete=False,
    ) as file:
        temporary = Path(file.name)
        file.write(text)
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def grid_centers(indices: np.ndarray, axis_min: float) -> np.ndarray:
    return axis_min + (indices.astype(np.float64) + 0.5) * GRID.voxel_size


def metric_index_bounds(low: float, high: float, axis_min: float, size: int) -> tuple[int, int]:
    start = max(0, int(np.floor((low - axis_min) / GRID.voxel_size)))
    stop = min(size, int(np.floor((high - axis_min) / GRID.voxel_size)) + 1)
    return start, stop


def voxelize_actor(semantics: np.ndarray, actor: dict[str, Any]) -> int:
    actor2ego = np.asarray(actor["actor2ego"], dtype=np.float64)
    ego2actor = np.linalg.inv(actor2ego)
    dimensions = np.asarray(actor["dimensions_xyz_m"], dtype=np.float64)
    corners = np.asarray(actor["corners_ego_m"], dtype=np.float64)
    half = dimensions / 2.0

    x0, x1 = metric_index_bounds(corners[:, 0].min(), corners[:, 0].max(), GRID.x_min, GRID.shape[0])
    y0, y1 = metric_index_bounds(corners[:, 1].min(), corners[:, 1].max(), GRID.y_min, GRID.shape[1])
    z0, z1 = metric_index_bounds(corners[:, 2].min(), corners[:, 2].max(), GRID.z_min, GRID.shape[2])
    if x0 >= x1 or y0 >= y1 or z0 >= z1:
        return 0

    ix, iy, iz = np.meshgrid(
        np.arange(x0, x1), np.arange(y0, y1), np.arange(z0, z1), indexing="ij"
    )
    points_ego = np.stack(
        [
            grid_centers(ix.ravel(), GRID.x_min),
            grid_centers(iy.ravel(), GRID.y_min),
            grid_centers(iz.ravel(), GRID.z_min),
            np.ones(ix.size, dtype=np.float64),
        ],
        axis=1,
    )
    points_actor = (points_ego @ ego2actor.T)[:, :3]
    inside = np.all(np.abs(points_actor) <= half + 1e-9, axis=1)
    if not np.any(inside):
        return 0
    semantics[ix.ravel()[inside], iy.ravel()[inside], iz.ravel()[inside]] = np.uint8(actor["occ3d_id"])
    return int(np.count_nonzero(inside))


def point_in_triangle_xy(points: np.ndarray, triangle: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a, b, c = triangle
    v0 = b[:2] - a[:2]
    v1 = c[:2] - a[:2]
    v2 = points - a[:2]
    denom = v0[0] * v1[1] - v1[0] * v0[1]
    if abs(denom) < 1e-12:
        return np.zeros(len(points), dtype=bool), np.zeros((len(points), 3), dtype=np.float64)
    u = (v2[:, 0] * v1[1] - v1[0] * v2[:, 1]) / denom
    v = (v0[0] * v2[:, 1] - v2[:, 0] * v0[1]) / denom
    w = 1.0 - u - v
    weights = np.stack([w, u, v], axis=1)
    return np.all(weights >= -1e-9, axis=1), weights


def voxelize_lane_ribbon(semantics: np.ndarray, lane: dict[str, Any]) -> int:
    vertices = np.asarray(lane["vertices_ego_m"], dtype=np.float64)
    written = 0
    for face in lane["triangle_indices"]:
        triangle = vertices[np.asarray(face, dtype=np.int64)]
        x0, x1 = metric_index_bounds(triangle[:, 0].min(), triangle[:, 0].max(), GRID.x_min, GRID.shape[0])
        y0, y1 = metric_index_bounds(triangle[:, 1].min(), triangle[:, 1].max(), GRID.y_min, GRID.shape[1])
        if x0 >= x1 or y0 >= y1:
            continue
        ix, iy = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1), indexing="ij")
        points = np.stack(
            [grid_centers(ix.ravel(), GRID.x_min), grid_centers(iy.ravel(), GRID.y_min)], axis=1
        )
        inside, weights = point_in_triangle_xy(points, triangle)
        if not np.any(inside):
            continue
        z_values = weights[inside] @ triangle[:, 2]
        iz = np.floor((z_values - GRID.z_min) / GRID.voxel_size).astype(np.int64)
        valid = (iz >= 0) & (iz < GRID.shape[2])
        if not np.any(valid):
            continue
        vx = ix.ravel()[inside][valid]
        vy = iy.ravel()[inside][valid]
        vz = iz[valid]
        target = semantics[vx, vy, vz]
        writable = target == UNKNOWN_ID
        semantics[vx[writable], vy[writable], vz[writable]] = np.uint8(11)
        written += int(np.count_nonzero(writable))
    return written


def voxelize_sample(source: dict[str, Any], keyframe: dict[str, Any], vector_map: dict[str, Any]) -> tuple[np.ndarray, dict[str, int]]:
    semantics = np.full(GRID.shape, UNKNOWN_ID, dtype=np.uint8)
    ego2map = np.asarray(keyframe["current_ego2global"], dtype=np.float64)
    lane_voxels = 0
    for raw_lane in vector_map.get("lanes", []):
        lane = normalize_lane(raw_lane, ego2map)
        if lane is not None:
            lane_voxels += voxelize_lane_ribbon(semantics, lane)
    actor_voxels = 0
    for actor in source["sources"]["actors"]:
        actor_voxels += voxelize_actor(semantics, actor)
    counts = Counter(int(value) for value in semantics.ravel())
    stats = {
        "lane_voxel_writes": lane_voxels,
        "actor_voxel_writes": actor_voxels,
        "unknown_voxels": counts[UNKNOWN_ID],
        "known_voxels": int(semantics.size - counts[UNKNOWN_ID]),
    }
    for class_id, count in sorted(counts.items()):
        stats[f"class_{class_id}_voxels"] = count
    return semantics, stats


def build_labels(
    *, source_manifest: Path, keyframe_manifest: Path, dataset_root: Path,
    output_root: Path, manifest_output: Path, summary_output: Path,
    limit: int | None, force: bool,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        if not force:
            raise FileExistsError(f"output exists: {output_root}")
        import shutil
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    source_rows = read_jsonl(source_manifest)
    keyframe_rows = read_jsonl(keyframe_manifest)
    # Keyframes are ordered by scene. Retain only the active VectorMap instead
    # of accumulating all 908 large JSON maps in memory. This keeps production
    # RSS bounded without changing any generated labels.
    active_map_path: Path | None = None
    active_vector_map: dict[str, Any] | None = None
    output_rows = []
    aggregate_classes = Counter()
    total_lane_writes = 0
    total_actor_writes = 0

    for index, (source, keyframe) in enumerate(zip(source_rows, keyframe_rows, strict=True), 1):
        if limit is not None and index > limit:
            break
        if source["sample_id"] != keyframe["sample_id"]:
            raise RuntimeError(f"manifest alignment mismatch at row {index}")
        map_path = dataset_root / source["map_source"]["relative_path"]
        if map_path != active_map_path:
            active_vector_map = json.loads(map_path.read_text(encoding="utf-8"))
            active_map_path = map_path
        assert active_vector_map is not None
        semantics, stats = voxelize_sample(source, keyframe, active_vector_map)
        sample_dir = output_root / source["scene_id"] / source["sample_id"]
        sample_dir.mkdir(parents=True, exist_ok=True)
        label_path = sample_dir / "labels.npz"
        mask_camera = np.zeros(GRID.shape, dtype=bool)
        mask_lidar = np.zeros(GRID.shape, dtype=bool)
        np.savez_compressed(label_path, semantics=semantics, mask_camera=mask_camera, mask_lidar=mask_lidar)
        class_counts = {str(value): count for value, count in Counter(int(v) for v in semantics.ravel()).items()}
        aggregate_classes.update({int(k): v for k, v in class_counts.items()})
        total_lane_writes += stats["lane_voxel_writes"]
        total_actor_writes += stats["actor_voxel_writes"]
        output_rows.append({
            "label_schema_version": LABEL_SCHEMA_VERSION,
            "voxelizer_version": VOXELIZER_VERSION,
            "sample_id": source["sample_id"],
            "scene_id": source["scene_id"],
            "labels_path": str(label_path.relative_to(output_root.parent.parent)),
            "semantics_shape": list(semantics.shape),
            "class_counts": class_counts,
            "lane_voxel_writes": stats["lane_voxel_writes"],
            "actor_voxel_writes": stats["actor_voxel_writes"],
            "mask_camera_policy": "all_false_pending_visibility_step",
            "mask_lidar_policy": "all_false_no_lidar",
        })
        if index == 1 or index % 100 == 0 or (limit is not None and index == limit):
            denominator = str(limit) if limit is not None else "all"
            print(f"[Step 6B] {index}/{denominator}: {source['sample_id']}")

    manifest_text = "".join(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n" for row in output_rows)
    summary = {
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "voxelizer_version": VOXELIZER_VERSION,
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256_file(source_manifest),
        "keyframe_manifest": str(keyframe_manifest),
        "keyframe_manifest_sha256": sha256_file(keyframe_manifest),
        "record_count": len(output_rows),
        "class_voxel_counts": {str(k): v for k, v in sorted(aggregate_classes.items())},
        "lane_voxel_writes_total": total_lane_writes,
        "actor_voxel_writes_total": total_actor_writes,
        "mask_camera_policy": "all_false_pending_visibility_step",
        "mask_lidar_policy": "all_false_no_lidar",
        "observed_free_generated": False,
        "vector_map_cache_policy": "active_scene_only",
        "trial": limit is not None,
        "manifest_sha256": hashlib.sha256(manifest_text.encode()).hexdigest(),
    }
    safe_write(manifest_output, manifest_text, force)
    safe_write(summary_output, json.dumps(summary, indent=2) + "\n", force)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, default=MANIFEST_ROOT / "occupancy_sources_v0.1.jsonl")
    parser.add_argument("--keyframe-manifest", type=Path, default=MANIFEST_ROOT / "occupancy_keyframes_v0.1.jsonl")
    parser.add_argument("--dataset-root", type=Path, default=ALPASIM_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTCOME_ROOT / "occupancy" / "labels_v0.1")
    parser.add_argument("--manifest-output", type=Path, default=MANIFEST_ROOT / "occupancy_labels_v0.1.jsonl")
    parser.add_argument("--summary-output", type=Path, default=REPORT_ROOT / "occupancy_label_summary_v0.1.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    summary = build_labels(
        source_manifest=args.source_manifest,
        keyframe_manifest=args.keyframe_manifest,
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        manifest_output=args.manifest_output,
        summary_output=args.summary_output,
        limit=args.limit,
        force=args.force,
    )
    print("Labels:", args.output_root)
    print("Manifest:", args.manifest_output)
    print("Summary:", args.summary_output)
    print("Records:", summary["record_count"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
