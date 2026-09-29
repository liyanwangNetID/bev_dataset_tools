"""Visual review for Step 6C per-camera conservative visibility masks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from project_paths import ALPASIM_DATA_ROOT, MANIFEST_ROOT, OUTCOME_ROOT, REPORT_ROOT
from step1.clip_manifest import CAMERA_NAMES
from step6.camera_visibility import (
    load_assignment_index,
    load_camera_contract,
    project_known_voxels,
)
from step6.contract import GRID, UNKNOWN_ID

COLORS = {
    0: (128, 128, 128), 3: (255, 128, 0), 4: (0, 0, 255),
    7: (255, 0, 255), 9: (0, 128, 255), 10: (0, 255, 255),
    11: (0, 180, 0), 255: (18, 18, 18),
}


def bev_projection(semantics: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    result = np.full(semantics.shape[:2], UNKNOWN_ID, dtype=np.uint8)
    selected = semantics != UNKNOWN_ID
    if mask is not None:
        selected &= mask
    road = np.any(selected & (semantics == 11), axis=2)
    result[road] = 11
    for class_id in np.unique(semantics):
        if class_id in (11, UNKNOWN_ID):
            continue
        result[np.any(selected & (semantics == class_id), axis=2)] = class_id
    return result


def colorize(labels: np.ndarray) -> np.ndarray:
    image = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for class_id in np.unique(labels):
        image[labels == class_id] = COLORS.get(int(class_id), (255,255,255))
    return image


def display_bev(labels: np.ndarray, title: str) -> np.ndarray:
    image = colorize(labels.T)
    image = cv2.flip(image, 0)
    image = cv2.resize(image, (400, 400), interpolation=cv2.INTER_NEAREST)
    origin_x = int(round((0.0 - GRID.x_min) / (GRID.x_max - GRID.x_min) * 400.0))
    origin_y = int(round((GRID.y_max - 0.0) / (GRID.y_max - GRID.y_min) * 400.0))
    cv2.drawMarker(image, (origin_x, origin_y), (255,255,255), cv2.MARKER_CROSS, 14, 2)
    header = np.zeros((42, 400, 3), dtype=np.uint8)
    cv2.putText(header, title, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255,255,255), 1, cv2.LINE_AA)
    return np.vstack([header, image])


def quadrant_counts(mask: np.ndarray) -> dict[str, int]:
    indices = np.argwhere(mask)
    if not len(indices):
        return {"front_left":0,"front_right":0,"rear_left":0,"rear_right":0}
    x = GRID.x_min + (indices[:,0] + 0.5) * GRID.voxel_size
    y = GRID.y_min + (indices[:,1] + 0.5) * GRID.voxel_size
    return {
        "front_left": int(np.count_nonzero((x >= 0) & (y >= 0))),
        "front_right": int(np.count_nonzero((x >= 0) & (y < 0))),
        "rear_left": int(np.count_nonzero((x < 0) & (y >= 0))),
        "rear_right": int(np.count_nonzero((x < 0) & (y < 0))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-root", type=Path, default=OUTCOME_ROOT / "review" / "step6c_visibility_labels_v0.1")
    parser.add_argument("--dataset-root", type=Path, default=ALPASIM_DATA_ROOT)
    parser.add_argument("--assignments", type=Path, default=MANIFEST_ROOT / "rectification_assignments_v0.1.jsonl")
    parser.add_argument("--rectification-root", type=Path, default=OUTCOME_ROOT / "calibration" / "rectification_v0.1")
    parser.add_argument("--output-root", type=Path, default=OUTCOME_ROOT / "review" / "step6c_visibility_visuals_v0.1")
    parser.add_argument("--report-output", type=Path, default=REPORT_ROOT / "step6c_visibility_review_v0.1.json")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.force:
        raise FileExistsError(f"output exists: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    assignments = load_assignment_index(args.assignments)
    paths = sorted(args.labels_root.rglob("labels.npz"))[:args.limit]
    records = []
    for index, label_path in enumerate(paths, 1):
        scene_id = label_path.parents[1].name
        sample_id = label_path.parent.name
        with np.load(label_path) as arrays:
            semantics = arrays["semantics"]
            combined_saved = arrays["mask_camera"]
        panels = [display_bev(bev_projection(semantics), "known semantics")]
        combined = np.zeros(GRID.shape, dtype=bool)
        camera_records = {}
        for camera_name in CAMERA_NAMES:
            extrinsic, k_rect, valid_mask, asset_id = load_camera_contract(
                scene_id, camera_name, args.dataset_root, assignments, args.rectification_root
            )
            camera_mask, stats = project_known_voxels(semantics, extrinsic, k_rect, valid_mask)
            combined |= camera_mask
            camera_records[camera_name] = {
                **stats,
                "asset_id": asset_id,
                "quadrants": quadrant_counts(camera_mask),
            }
            panels.append(display_bev(bev_projection(semantics, camera_mask), camera_name))
        if not np.array_equal(combined, combined_saved):
            raise RuntimeError(f"recomputed combined mask differs: {sample_id}")
        top = np.hstack(panels[:3])
        bottom = np.hstack(panels[3:])
        pad = top.shape[1] - bottom.shape[1]
        if pad > 0:
            bottom = cv2.copyMakeBorder(bottom, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(0,0,0))
        montage = np.vstack([top, bottom])
        output = args.output_root / scene_id / f"{sample_id}.jpg"
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), montage, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
        records.append({
            "sample_id": sample_id,
            "scene_id": scene_id,
            "visual_path": str(output),
            "combined_visible": int(combined.sum()),
            "cameras": camera_records,
        })
        print(f"[Step 6C Review] {index}/{len(paths)}: {output}")
    report = {"record_count": len(records), "axis_contract": "x right, y up, white cross current ego", "records": records}
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Visuals:", args.output_root)
    print("Report:", args.report_output)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
