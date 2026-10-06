from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
import torch
from .calibration import CalibrationResolver
from .constants import CAMERA_ORDER, OCCUPANCY_SHAPE, OUTCOME_ROOT, RAW_DATA_ROOT, UNKNOWN_LABEL
from .manifests import ManifestIndex


def _first(record, *keys):
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def _camera_path_value(value, camera=None):
    """Return an image path from the frozen manifest camera record.

    Formal manifests store each camera as a dictionary containing the path and
    timing metadata. Older probes and synthetic tests may provide a path
    string directly, so both forms remain supported.
    """
    if isinstance(value, (str, bytes, Path)):
        return value
    if not isinstance(value, dict):
        raise TypeError(
            f"camera {camera or '<unknown>'}: expected path or camera record, "
            f"got {type(value).__name__}"
        )
    for key in (
        "image_path", "data_path", "path", "relative_path", "file_path",
        "filename", "image", "file",
    ):
        candidate = value.get(key)
        if isinstance(candidate, (str, bytes, Path)):
            return candidate
    path_like = [candidate for candidate in value.values()
                 if isinstance(candidate, str) and
                 candidate.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
    if len(path_like) == 1:
        return path_like[0]
    raise KeyError(
        f"camera {camera or '<unknown>'}: no unique image path in record; "
        f"keys={sorted(value)}"
    )


def _resolve(root, value, camera=None):
    path = Path(_camera_path_value(value, camera=camera)).expanduser()
    return path if path.is_absolute() else Path(root) / path


def _camera_paths(keyframe, history=False):
    keys = ("history_cameras", "history_camera_paths") if history else ("current_cameras", "camera_paths", "cameras")
    value = _first(keyframe, *keys)
    if not isinstance(value, dict):
        raise KeyError(f"camera path dictionary not found for history={history}")
    return value


class Step7SampleLoader:
    def __init__(self, index=None, raw_root=RAW_DATA_ROOT, outcome_root=OUTCOME_ROOT):
        self.index = index or ManifestIndex()
        self.raw_root = Path(raw_root)
        self.outcome_root = Path(outcome_root)
        self.calibration = CalibrationResolver(raw_root=raw_root)

    def _load_frame(self, paths, calibration):
        raw_images, images, valid_masks = [], [], []
        for camera in CAMERA_ORDER:
            image_path = _resolve(self.raw_root, paths[camera], camera=camera)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(image_path)
            with np.load(calibration[camera]["maps_path"], allow_pickle=False) as maps:
                rectified = cv2.remap(image, maps["map_x"], maps["map_y"], cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT)
                valid = maps["valid_mask"].astype(bool)
            raw_images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            images.append(cv2.cvtColor(rectified, cv2.COLOR_BGR2RGB))
            valid_masks.append(valid)
        return np.stack(raw_images), np.stack(images), np.stack(valid_masks)

    def load(self, sample_id):
        joined = self.index.get(sample_id)
        keyframe, label = joined["keyframe"], joined["label"]
        clip_id = _first(keyframe, "clip_id", "scene_name") or sample_id.rsplit("_", 1)[0]
        calibration_ids = keyframe["calibration_ids"]
        calibration = self.calibration.resolve(clip_id, calibration_ids)
        current_raw, current, current_valid = self._load_frame(_camera_paths(keyframe), calibration)
        try:
            history_paths = _camera_paths(keyframe, history=True)
        except KeyError:
            history_paths = _camera_paths(keyframe)
        history_raw, history, history_valid = self._load_frame(history_paths, calibration)
        label_path_value = _first(label, "labels_path", "label_path", "occ_path", "path")
        if label_path_value is None:
            raise KeyError(f"{sample_id}: label path is missing")
        label_path = _resolve(self.outcome_root, label_path_value)
        with np.load(label_path, allow_pickle=False) as payload:
            semantics = payload["semantics"].astype(np.int64)
            mask_camera = payload["mask_camera"].astype(bool)
            mask_lidar = payload["mask_lidar"].astype(bool)
        if semantics.shape != OCCUPANCY_SHAPE:
            raise ValueError(f"semantics shape {semantics.shape} != {OCCUPANCY_SHAPE}")
        target = semantics.copy()
        target[~mask_camera] = UNKNOWN_LABEL
        camera_to_ego = np.stack([calibration[c]["camera_to_ego"] for c in CAMERA_ORDER])
        intrins = np.stack([calibration[c]["K_rect"] for c in CAMERA_ORDER])
        sample = {
            "sample_id": sample_id, "clip_id": clip_id,
            "camera_order": CAMERA_ORDER,
            "raw_images_uint8": np.stack([current_raw, history_raw]),
            "raw_images_uint8": np.stack([current_raw, history_raw]),
            "images_uint8": np.stack([current, history]),
            "image_valid_masks": np.stack([current_valid, history_valid]),
            "imgs": torch.from_numpy(np.stack([current, history])).permute(0, 1, 4, 2, 3).float() / 255.0,
            "rots": torch.from_numpy(camera_to_ego[:, :3, :3]).repeat(2, 1, 1, 1),
            "trans": torch.from_numpy(camera_to_ego[:, :3, 3]).repeat(2, 1, 1),
            "intrins": torch.from_numpy(intrins).repeat(2, 1, 1, 1),
            "post_rots": torch.eye(3).repeat(2, len(CAMERA_ORDER), 1, 1),
            "post_trans": torch.zeros(2, len(CAMERA_ORDER), 3),
            "bda": torch.eye(3),
            "gt_occupancy": torch.from_numpy(target),
            "gt_occupancy_ori": torch.from_numpy(semantics),
            "mask_camera": torch.from_numpy(mask_camera),
            "mask_lidar": torch.from_numpy(mask_lidar),
            "metadata": {"calibration": calibration, "keyframe": keyframe, "label": label,
                         "is_scene_start": bool(keyframe.get("is_scene_start", False))},
        }
        return sample
