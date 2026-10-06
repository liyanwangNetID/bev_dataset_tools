from __future__ import annotations


def to_occstudio_batch(batch):
    """Return the thin bridge expected by OccStudio-style train entrypoints.

    Time order in every tensor is [current, history]. Camera order is the
    frozen four-camera order stored in metadata.
    """
    required = (
        "imgs", "rots", "trans", "intrins", "post_rots", "post_trans",
        "bda", "gt_occupancy", "mask_camera", "mask_lidar", "metadata",
    )
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError(f"batch missing keys: {missing}")
    img_inputs = (
        batch["imgs"], batch["rots"], batch["trans"], batch["intrins"],
        batch["post_rots"], batch["post_trans"], batch["bda"],
    )
    img_metas = []
    for metadata in batch["metadata"]:
        keyframe = metadata["keyframe"]
        img_metas.append({
            "sample_id": keyframe["sample_id"],
            "scene_id": keyframe.get("scene_id") or keyframe.get("clip_id"),
            "start_of_sequence": bool(keyframe.get("is_scene_start", False)),
            "history_is_repeated_current": bool(
                keyframe.get("history_is_repeated_current", False)
            ),
            "history_delta_ns": int(keyframe.get("history_delta_ns", 0)),
            "camera_order": tuple(keyframe["camera_order"]),
        })
    return {
        "img_inputs": img_inputs,
        "gt_occupancy": batch["gt_occupancy"],
        "mask_camera": batch["mask_camera"],
        "mask_lidar": batch["mask_lidar"],
        "img_metas": img_metas,
    }
