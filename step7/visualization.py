from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np


def _text(image, value, xy=(12, 28), scale=0.66):
    cv2.putText(image, value, xy, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, value, xy, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 1, cv2.LINE_AA)


def camera_mosaic(images, names, title, panel_size=(640, 360)):
    panels = []
    for image, name in zip(images, names):
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        bgr = cv2.resize(bgr, panel_size, interpolation=cv2.INTER_AREA)
        _text(bgr, name)
        panels.append(bgr)
    mosaic = np.concatenate([
        np.concatenate(panels[:2], axis=1),
        np.concatenate(panels[2:], axis=1),
    ], axis=0)
    _text(mosaic, title, (12, mosaic.shape[0] - 14))
    return mosaic


def _palette():
    palette = np.random.default_rng(7).integers(35, 240, (256, 3), dtype=np.uint8)
    palette[17] = (45, 45, 45)
    palette[255] = (255, 255, 255)
    return palette


def occupancy_bev(semantics, mask):
    valid = np.where(mask, semantics, 255)
    bev = np.full(valid.shape[:2], 255, np.uint8)
    for z in range(valid.shape[2]):
        layer = valid[:, :, z]
        occupied = (layer != 17) & (layer != 255)
        bev[occupied] = layer[occupied]
        free = (layer == 17) & (bev == 255)
        bev[free] = 17
    return cv2.resize(_palette()[bev], (800, 800), interpolation=cv2.INTER_NEAREST)


def mask_bev(mask):
    count = mask.sum(axis=2).astype(np.float32)
    image = np.clip(count / max(mask.shape[2], 1) * 255, 0, 255).astype(np.uint8)
    image = cv2.applyColorMap(image, cv2.COLORMAP_VIRIDIS)
    return cv2.resize(image, (800, 800), interpolation=cv2.INTER_NEAREST)


def height_slices(semantics, mask):
    palette = _palette()
    panels = []
    for z in range(semantics.shape[2]):
        layer = np.where(mask[:, :, z], semantics[:, :, z], 255).astype(np.uint8)
        image = cv2.resize(palette[layer], (240, 240), interpolation=cv2.INTER_NEAREST)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        _text(image, f"z={z}", (8, 22), 0.55)
        panels.append(image)
    rows = [np.concatenate(panels[i:i + 4], axis=1) for i in range(0, 16, 4)]
    return np.concatenate(rows, axis=0)


def save_validation(sample, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    names = sample["camera_order"]
    raw = sample["raw_images_uint8"]
    frames = sample["images_uint8"]
    for index, frame_name in enumerate(("current", "history")):
        cv2.imwrite(str(output / f"{frame_name}_raw_cameras.png"),
                    camera_mosaic(raw[index], names, f"{frame_name} raw"))
        cv2.imwrite(str(output / f"{frame_name}_rectified_cameras.png"),
                    camera_mosaic(frames[index], names, f"{frame_name} rectified"))
    semantics = sample["gt_occupancy_ori"].numpy()
    mask = sample["mask_camera"].numpy()
    bev = occupancy_bev(semantics, mask)
    cv2.imwrite(str(output / "occupancy_semantics_bev.png"), cv2.cvtColor(bev, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output / "occupancy_camera_mask_bev.png"), mask_bev(mask))
    cv2.imwrite(str(output / "occupancy_height_slices.png"), height_slices(semantics, mask))

    summary = {}
    for key in ("imgs", "rots", "trans", "intrins", "post_rots", "post_trans", "bda",
                "gt_occupancy", "mask_camera", "mask_lidar"):
        value = sample[key]
        summary[key] = {"shape": list(value.shape), "dtype": str(value.dtype),
                        "min": float(value.min()), "max": float(value.max())}
    labels, counts = np.unique(semantics, return_counts=True)
    summary["semantic_histogram"] = {str(int(k)): int(v) for k, v in zip(labels, counts)}
    summary["camera_visible_voxels"] = int(mask.sum())
    summary["camera_visible_ratio"] = float(mask.mean())
    summary.update({"sample_id": sample["sample_id"], "clip_id": sample["clip_id"],
                    "camera_order": list(names), "metadata": sample["metadata"]})
    (output / "sample_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    np.savez_compressed(output / "tensors.npz",
                        gt_occupancy=sample["gt_occupancy"].numpy(),
                        mask_camera=mask, intrins=sample["intrins"].numpy(),
                        rots=sample["rots"].numpy(), trans=sample["trans"].numpy())
