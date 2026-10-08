"""Step 6C conservative camera visibility for known semantic voxels.

Only voxels with a confirmed semantic label are candidates. A voxel is visible
when its center projects inside a rectified camera valid mask and survives a
per-camera z-buffer over known labeled voxels. Unknown voxels remain unmarked;
this step does not synthesize observed-free space.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from project_paths import ALPASIM_DATA_ROOT, MANIFEST_ROOT, OUTCOME_ROOT, REPORT_ROOT
from step1.clip_manifest import CAMERA_NAMES
from step2.calibration import read_camera_calibration
from step6.contract import GRID, OBSERVED_FREE_ID, UNKNOWN_ID
from step6.source_geometry import quaternion_matrix

VISIBILITY_VERSION = "0.2.0"
TARGET_WIDTH = 960
TARGET_HEIGHT = 540


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def load_assignment_index(path: Path) -> dict[tuple[str, str], str]:
    index = {}
    for row in read_jsonl(path):
        key = (row["clip_id"], row["camera_name"])
        if key in index:
            raise ValueError(f"duplicate rectification assignment: {key}")
        index[key] = row["rectification_asset_id"]
    return index


def rigid_matrix_from_calibration(calibration) -> np.ndarray:
    q = calibration.rig_to_camera_quaternion
    t = calibration.rig_to_camera_translation
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion_matrix({"x": q.x, "y": q.y, "z": q.z, "w": q.w})
    matrix[:3, 3] = [t.x, t.y, t.z]
    # Recorded values describe the camera pose in the rig frame. Projection
    # needs rig/current-ego points in the optical camera frame, so invert it.
    return np.linalg.inv(matrix)


def voxel_centers(indices: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            GRID.x_min + (indices[:, 0] + 0.5) * GRID.voxel_size,
            GRID.y_min + (indices[:, 1] + 0.5) * GRID.voxel_size,
            GRID.z_min + (indices[:, 2] + 0.5) * GRID.voxel_size,
        ],
        axis=1,
    )


def project_voxel_contract(
    semantics: np.ndarray,
    rig_to_camera: np.ndarray,
    k_rect: np.ndarray,
    valid_pixel_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Project all voxel centers and derive frustum, supervision, and free masks.

    ``frustum`` contains every voxel whose center projects through a valid
    rectified pixel with positive camera depth. Unknown voxels in this region
    become semantic ID 17. ``supervised`` is deliberately narrower: for pixels
    with a known occupied surface, it contains voxels no farther than the first
    occupied surface, plus that first surface. This makes any free voxel behind
    the first known surface ignorable with mask_camera.
    """
    all_indices = np.indices(GRID.shape, dtype=np.int16).reshape(3, -1).T
    points = voxel_centers(all_indices)
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    camera = (homogeneous @ np.asarray(rig_to_camera, dtype=np.float64).T)[:, :3]
    depth = camera[:, 2]
    positive = depth > 1e-6

    frustum_flat = np.zeros(len(all_indices), dtype=bool)
    source_ids = np.flatnonzero(positive)
    if len(source_ids):
        cam = camera[source_ids]
        uvw = cam @ np.asarray(k_rect, dtype=np.float64).T
        px = np.rint(uvw[:, 0] / uvw[:, 2]).astype(np.int64)
        py = np.rint(uvw[:, 1] / uvw[:, 2]).astype(np.int64)
        inside = (px >= 0) & (px < TARGET_WIDTH) & (py >= 0) & (py < TARGET_HEIGHT)
        valid_local = np.flatnonzero(inside)
        if len(valid_local):
            valid_local = valid_local[valid_pixel_mask[py[valid_local], px[valid_local]]]
        source_ids = source_ids[valid_local]
        px = px[valid_local]
        py = py[valid_local]
        frustum_flat[source_ids] = True
    else:
        px = np.empty(0, dtype=np.int64)
        py = np.empty(0, dtype=np.int64)

    frustum = frustum_flat.reshape(GRID.shape)
    occupied_flat = (semantics.reshape(-1) != UNKNOWN_ID) & (semantics.reshape(-1) != OBSERVED_FREE_ID)
    known_local = np.flatnonzero(occupied_flat[source_ids])
    supervised_flat = np.zeros(len(all_indices), dtype=bool)
    first_surface_flat = np.zeros(len(all_indices), dtype=bool)

    if len(known_local):
        known_source = source_ids[known_local]
        known_pixel = py[known_local] * TARGET_WIDTH + px[known_local]
        known_depth = depth[known_source]
        order = np.lexsort((known_depth, known_pixel))
        sorted_pixels = known_pixel[order]
        first = np.ones(len(order), dtype=bool)
        first[1:] = sorted_pixels[1:] != sorted_pixels[:-1]
        winner_local = known_local[order[first]]
        winner_source = source_ids[winner_local]
        winner_pixels = py[winner_local] * TARGET_WIDTH + px[winner_local]
        winner_depths = depth[winner_source]
        first_surface_flat[winner_source] = True

        all_pixel = py * TARGET_WIDTH + px
        surface_depth = np.full(TARGET_WIDTH * TARGET_HEIGHT, np.inf, dtype=np.float32)
        surface_depth[winner_pixels] = winner_depths.astype(np.float32)
        ray_has_surface = np.isfinite(surface_depth[all_pixel])
        supervised_local = ray_has_surface & (
            depth[source_ids] <= surface_depth[all_pixel] + GRID.voxel_size * 0.75
        )
        supervised_flat[source_ids[supervised_local]] = True

    free = frustum & (semantics == UNKNOWN_ID)
    supervised = supervised_flat.reshape(GRID.shape)
    first_surface = first_surface_flat.reshape(GRID.shape)
    stats = {
        'grid_voxels': int(np.prod(GRID.shape)),
        'positive_depth': int(positive.sum()),
        'valid_frustum': int(frustum.sum()),
        'observed_free_candidates': int(free.sum()),
        'first_surface': int(first_surface.sum()),
        'supervised': int(supervised.sum()),
    }
    return frustum, free, supervised, stats


def project_known_voxels(
    semantics: np.ndarray,
    rig_to_camera: np.ndarray,
    k_rect: np.ndarray,
    valid_pixel_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    """Compatibility wrapper returning the new supervised camera mask."""
    _, _, supervised, stats = project_voxel_contract(
        semantics, rig_to_camera, k_rect, valid_pixel_mask
    )
    return supervised, {
        'known': int(np.count_nonzero((semantics != UNKNOWN_ID) & (semantics != OBSERVED_FREE_ID))),
        'positive_depth': stats['positive_depth'],
        'valid_projection': stats['valid_frustum'],
        'zbuffer_visible': stats['first_surface'],
        'supervised': stats['supervised'],
        'observed_free_candidates': stats['observed_free_candidates'],
    }

def load_camera_contract(
    scene_id: str,
    camera_name: str,
    dataset_root: Path,
    assignment_index: dict[tuple[str, str], str],
    rectification_root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    asset_id = assignment_index[(scene_id, camera_name)]
    asset_dir = rectification_root / "intrinsics_assets" / asset_id
    metadata = json.loads((asset_dir / "metadata.json").read_text(encoding="utf-8"))
    valid_mask = cv2.imread(str(asset_dir / "valid_mask.png"), cv2.IMREAD_GRAYSCALE)
    if valid_mask is None or valid_mask.shape != (TARGET_HEIGHT, TARGET_WIDTH):
        raise ValueError(f"invalid rectification valid mask: {asset_id}")
    calibration = read_camera_calibration(
        dataset_root / scene_id / "calibration" / f"{camera_name}.json",
        camera_name,
    )
    return (
        rigid_matrix_from_calibration(calibration),
        np.asarray(metadata["K_rect"], dtype=np.float64),
        valid_mask > 0,
        asset_id,
    )


def update_label(
    label_path: Path,
    scene_id: str,
    dataset_root: Path,
    assignment_index: dict[tuple[str, str], str],
    rectification_root: Path,
) -> dict[str, Any]:
    with np.load(label_path) as arrays:
        semantics = arrays["semantics"]
        mask_lidar = arrays["mask_lidar"]
    combined = np.zeros(GRID.shape, dtype=bool)
    camera_counts = {}
    assets = {}
    for camera_name in CAMERA_NAMES:
        extrinsic, k_rect, valid_mask, asset_id = load_camera_contract(
            scene_id, camera_name, dataset_root, assignment_index, rectification_root
        )
        camera_visible, stats = project_known_voxels(semantics, extrinsic, k_rect, valid_mask)
        combined |= camera_visible
        camera_counts[camera_name] = stats
        assets[camera_name] = asset_id
    np.savez_compressed(
        label_path,
        semantics=semantics,
        mask_camera=combined,
        mask_lidar=mask_lidar,
    )
    known = semantics != UNKNOWN_ID
    return {
        "mask_camera_true": int(combined.sum()),
        "known_voxels": int(known.sum()),
        "visible_known_ratio": float(combined.sum() / known.sum()) if known.any() else 0.0,
        "camera_projection_counts": camera_counts,
        "rectification_assets": assets,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-manifest", type=Path, default=MANIFEST_ROOT / "occupancy_labels_v0.1.jsonl")
    parser.add_argument("--dataset-root", type=Path, default=ALPASIM_DATA_ROOT)
    parser.add_argument("--outcome-root", type=Path, default=OUTCOME_ROOT)
    parser.add_argument("--assignments", type=Path, default=MANIFEST_ROOT / "rectification_assignments_v0.1.jsonl")
    parser.add_argument("--rectification-root", type=Path, default=OUTCOME_ROOT / "calibration" / "rectification_v0.1")
    parser.add_argument("--report-output", type=Path, default=REPORT_ROOT / "step6c_camera_visibility_trial_v0.1.json")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--copy-root", type=Path, default=OUTCOME_ROOT / "review" / "step6c_visibility_labels_v0.1")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.copy_root.exists():
        if not args.force and any(args.copy_root.iterdir()):
            raise FileExistsError(f"output exists: {args.copy_root}")
        if args.force:
            shutil.rmtree(args.copy_root)
    args.copy_root.mkdir(parents=True, exist_ok=True)

    assignments = load_assignment_index(args.assignments)
    records = []
    aggregate_camera = Counter()
    for index, row in enumerate(read_jsonl(args.label_manifest), 1):
        if index > args.limit:
            break
        source = args.outcome_root / row["labels_path"]
        target = args.copy_root / row["scene_id"] / row["sample_id"] / "labels.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        stats = update_label(
            target, row["scene_id"], args.dataset_root, assignments, args.rectification_root
        )
        aggregate_camera.update({name: value["zbuffer_visible"] for name, value in stats["camera_projection_counts"].items()})
        records.append({"sample_id": row["sample_id"], "scene_id": row["scene_id"], "labels_path": str(target), **stats})
        print(f"[Step 6C] {index}/{args.limit}: {row['sample_id']} visible={stats['mask_camera_true']}")

    report = {
        "visibility_version": VISIBILITY_VERSION,
        "record_count": len(records),
        "policy": "known_semantic_voxel_centers_only + rectified_valid_mask + per_camera_zbuffer",
        "unknown_voxels_visible": False,
        "observed_free_generated": False,
        "aggregate_camera_zbuffer_visible": dict(aggregate_camera),
        "records": records,
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Labels:", args.copy_root)
    print("Report:", args.report_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
