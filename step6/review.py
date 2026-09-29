"""Generate deterministic visual and numeric review artifacts for Step 6B labels."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from project_paths import OUTCOME_ROOT, REPORT_ROOT
from step6.contract import GRID, OCC3D_CLASSES, UNKNOWN_ID

COLORS_BGR = {
    0: (128, 128, 128),
    3: (255, 128, 0),
    4: (0, 0, 255),
    7: (255, 0, 255),
    9: (0, 128, 255),
    10: (0, 255, 255),
    11: (0, 180, 0),
    255: (20, 20, 20),
}


def colorize(labels: np.ndarray) -> np.ndarray:
    image = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for class_id in np.unique(labels):
        image[labels == class_id] = COLORS_BGR.get(int(class_id), (255, 255, 255))
    return image


def bev_projection(semantics: np.ndarray) -> np.ndarray:
    """Project top-most known voxel; actors have priority over road."""
    result = np.full(semantics.shape[:2], UNKNOWN_ID, dtype=np.uint8)
    known = semantics != UNKNOWN_ID
    road = np.any(semantics == 11, axis=2)
    result[road] = 11
    actor_ids = [value for value in np.unique(semantics) if value not in (11, UNKNOWN_ID)]
    for class_id in actor_ids:
        result[np.any(semantics == class_id, axis=2)] = class_id
    return result


def add_header(image: np.ndarray, text: str) -> np.ndarray:
    header = np.zeros((42, image.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255,255,255), 1, cv2.LINE_AA)
    return np.vstack([header, image])


def render_one(label_path: Path, output_path: Path) -> dict:
    with np.load(label_path) as data:
        semantics = data["semantics"]
        mask_camera = data["mask_camera"]
        mask_lidar = data["mask_lidar"]
    if semantics.shape != GRID.shape or semantics.dtype != np.uint8:
        raise ValueError(f"invalid semantics contract: {label_path}")
    # Array axis order is [x, y, z]. Images use [row, column], so transpose
    # the BEV plane to display x horizontally and y vertically. Flip image rows
    # so positive y points upward. This matches the z-slice convention below.
    bev_labels = bev_projection(semantics).T
    bev = colorize(bev_labels)
    bev = cv2.flip(bev, 0)
    bev = cv2.resize(bev, (800, 800), interpolation=cv2.INTER_NEAREST)

    # Mark the current-ego origin at x=0, y=0. The marker is review-only and
    # never modifies labels.npz.
    origin_x = int(round((0.0 - GRID.x_min) / (GRID.x_max - GRID.x_min) * 800.0))
    origin_y = int(round((GRID.y_max - 0.0) / (GRID.y_max - GRID.y_min) * 800.0))
    cv2.drawMarker(
        bev,
        (origin_x, origin_y),
        (255, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=18,
        thickness=2,
    )
    bev = add_header(bev, "BEV: x right, y up; white cross=ego; dark=unknown")

    slices = []
    for z_index in range(GRID.shape[2]):
        z0 = GRID.z_min + z_index * GRID.voxel_size
        plane = colorize(semantics[:, :, z_index].T)
        plane = cv2.flip(plane, 0)
        plane = cv2.resize(plane, (200, 200), interpolation=cv2.INTER_NEAREST)
        slices.append(add_header(plane, f"z[{z_index}] {z0:.1f}..{z0+GRID.voxel_size:.1f}m"))
    slice_grid = np.vstack([np.hstack(slices[i:i+4]) for i in range(0, 16, 4)])

    # Preserve the square BEV geometry. The slice panel has four row headers,
    # so pad the shorter BEV panel vertically instead of stretching it.
    if bev.shape[0] > slice_grid.shape[0]:
        raise ValueError(
            f"BEV panel taller than slice grid: {bev.shape[0]} > {slice_grid.shape[0]}"
        )
    pad_height = slice_grid.shape[0] - bev.shape[0]
    if pad_height:
        bev = cv2.copyMakeBorder(
            bev,
            0,
            pad_height,
            0,
            0,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

    montage = np.hstack([bev, slice_grid])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), montage, [int(cv2.IMWRITE_JPEG_QUALITY), 94])

    values, counts = np.unique(semantics, return_counts=True)
    per_z = []
    for z_index in range(GRID.shape[2]):
        z_values, z_counts = np.unique(semantics[:, :, z_index], return_counts=True)
        per_z.append({str(int(k)): int(v) for k, v in zip(z_values, z_counts)})
    return {
        "label_path": str(label_path),
        "review_path": str(output_path),
        "class_counts": {str(int(k)): int(v) for k, v in zip(values, counts)},
        "per_z_class_counts": per_z,
        "mask_camera_true": int(mask_camera.sum()),
        "mask_lidar_true": int(mask_lidar.sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-root", type=Path, default=OUTCOME_ROOT / "review" / "step6b_trial_labels_v0.1")
    parser.add_argument("--output-root", type=Path, default=OUTCOME_ROOT / "review" / "step6b_trial_visuals_v0.1")
    parser.add_argument("--report-output", type=Path, default=REPORT_ROOT / "step6b_trial_review_v0.1.json")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    paths = sorted(args.labels_root.rglob("labels.npz"))[:args.limit]
    if not paths:
        raise FileNotFoundError(f"no labels.npz under {args.labels_root}")
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.force:
        raise FileExistsError(f"output exists: {args.output_root}")
    records = []
    aggregate = Counter()
    for index, path in enumerate(paths, 1):
        relative = path.relative_to(args.labels_root).with_suffix(".jpg")
        output = args.output_root / relative
        record = render_one(path, output)
        records.append(record)
        aggregate.update({int(k): v for k, v in record["class_counts"].items()})
        print(f"[Step 6B Review] {index}/{len(paths)}: {output}")
    report = {
        "record_count": len(records),
        "class_names": {str(k): v for k, v in OCC3D_CLASSES.items()},
        "aggregate_class_counts": {str(k): v for k, v in sorted(aggregate.items())},
        "records": records,
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Visuals:", args.output_root)
    print("Report:", args.report_output)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
