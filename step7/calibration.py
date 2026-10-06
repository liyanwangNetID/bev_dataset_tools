from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from .constants import CAMERA_ORDER, RAW_DATA_ROOT, RECTIFICATION_ROOT


def canonical_hash(value):
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _walk(value):
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _quaternion_wxyz_to_rotation(quat):
    values = np.asarray([quat[key] for key in ("w", "x", "y", "z")], dtype=np.float64)
    norm = np.linalg.norm(values)
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError(f"invalid quaternion: {quat}")
    w, x, y, z = values / norm
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def _rig_to_camera_matrix(raw):
    """Parse the frozen AlpaSim ``available_camera.rig_to_camera`` contract.

    The JSON names the transform direction explicitly. The stored quaternion
    and vector therefore form T_camera_from_rig. OccStudio requires
    T_ego_from_camera, and the AlpaSim rig frame is the ego frame, so the
    returned matrix is the inverse of the stored transform.
    """
    try:
        value = raw["available_camera"]["rig_to_camera"]
        quat = value["quat"]
        vec = value["vec"]
    except (KeyError, TypeError) as exc:
        raise ValueError("missing available_camera.rig_to_camera.quat/vec") from exc
    rig_to_camera = np.eye(4, dtype=np.float32)
    rig_to_camera[:3, :3] = _quaternion_wxyz_to_rotation(quat)
    rig_to_camera[:3, 3] = np.asarray([vec[key] for key in ("x", "y", "z")], dtype=np.float32)
    camera_to_ego = np.linalg.inv(rig_to_camera).astype(np.float32)
    rotation = camera_to_ego[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5):
        raise ValueError("camera_to_ego rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError("camera_to_ego rotation determinant is not +1")
    return camera_to_ego


def _matrix_from_node(node):
    if isinstance(node, list):
        array = np.asarray(node, dtype=np.float32)
        if array.shape == (4, 4):
            return array
        if array.size == 16:
            return array.reshape(4, 4)
    if not isinstance(node, dict):
        return None
    for key in ("matrix", "transform", "camera_to_ego", "cam2ego", "sensor_to_ego",
                "T_camera_ego", "T_cam_ego"):
        if key in node:
            result = _matrix_from_node(node[key])
            if result is not None:
                return result
    translation = node.get("translation") or node.get("position")
    rotation = node.get("rotation") or node.get("quaternion")
    if isinstance(translation, dict):
        translation = [translation.get(k, 0.0) for k in ("x", "y", "z")]
    if isinstance(rotation, dict):
        rotation = [rotation.get(k) for k in ("w", "x", "y", "z")]
    if translation is not None and rotation is not None and None not in rotation:
        w, x, y, z = map(float, rotation)
        norm = np.linalg.norm([w, x, y, z])
        w, x, y, z = np.asarray([w, x, y, z]) / norm
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                          [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                          [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
        matrix[:3, 3] = np.asarray(translation, dtype=np.float32)
        return matrix
    return None


class CalibrationResolver:
    def __init__(self, raw_root=RAW_DATA_ROOT, rectification_root=RECTIFICATION_ROOT):
        self.raw_root = Path(raw_root)
        self.rectification_root = Path(rectification_root)
        self._assets = []
        for metadata_path in sorted((self.rectification_root / "intrinsics_assets").glob("*/metadata.json")):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["metadata_path"] = str(metadata_path)
            metadata["maps_path"] = str(metadata_path.parent / "maps.npz")
            self._assets.append(metadata)

    def _raw(self, clip_id, camera):
        path = self.raw_root / clip_id / "calibration" / f"{camera}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        return path, json.loads(path.read_text(encoding="utf-8"))

    def _intrinsic_candidates(self, raw, logical_id):
        candidates = []
        for node in _walk(raw):
            if isinstance(node, dict) and node.get("logical_id") == logical_id:
                if "intrinsics" in node:
                    candidates.append({"logical_id": logical_id, "intrinsics": node["intrinsics"]})
                candidates.append(node)
        return candidates

    def resolve(self, clip_id, calibration_ids):
        result = {}
        for camera in CAMERA_ORDER:
            logical_id = calibration_ids[camera]
            path, raw = self._raw(clip_id, camera)
            candidates = self._intrinsic_candidates(raw, logical_id)
            candidate_hashes = {canonical_hash(candidate) for candidate in candidates}
            matching = []
            for asset in self._assets:
                if asset["camera_name"] != camera:
                    continue
                source = asset["source_intrinsics"]
                # Match the raw clip calibration against the frozen source object.
                # Do not hash the asset and compare it with its own stored hash.
                if source in candidates or asset["intrinsic_hash"] in candidate_hashes:
                    matching.append(asset)
            if len(matching) != 1:
                diagnostic = {
                    "logical_id": logical_id,
                    "candidate_hashes": sorted(candidate_hashes),
                    "camera_assets": [
                        {"asset_id": asset["asset_id"], "intrinsic_hash": asset["intrinsic_hash"]}
                        for asset in self._assets if asset["camera_name"] == camera
                    ],
                }
                raise ValueError(
                    f"{clip_id}/{camera}: expected one rectification asset, got {len(matching)}; "
                    f"diagnostic={diagnostic}"
                )
            try:
                extrinsic = _rig_to_camera_matrix(raw)
            except ValueError:
                extrinsic = None
                for node in _walk(raw):
                    extrinsic = _matrix_from_node(node)
                    if extrinsic is not None:
                        break
                if extrinsic is None:
                    raise ValueError(f"{path}: camera_to_ego matrix was not found")
            asset = matching[0]
            result[camera] = {
                "calibration_path": str(path), "asset_id": asset["asset_id"],
                "maps_path": asset["maps_path"], "K_rect": np.asarray(asset["K_rect"], np.float32),
                "camera_to_ego": extrinsic, "valid_pixel_ratio": asset["valid_pixel_ratio"],
            }
        return result
