"""Strict reader for per-Clip single-interval 2.5D static occupancy.

The source grid is in the AlpaSim map frame. Grid cell (i, j) covers
[i*s, (i+1)*s) x [j*s, (j+1)*s), where s is grid_size_xy_m.
The reader preserves source arrays and does not assign semantic labels.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_ARRAYS = {
    "grid_i": np.dtype("int32"),
    "grid_j": np.dtype("int32"),
    "min_z": np.dtype("float32"),
    "max_z": np.dtype("float32"),
    "point_count": np.dtype("uint32"),
    "flags": np.dtype("uint16"),
}
EXPECTED_REPRESENTATION = "single_interval_2p5d_static_occupancy_road_filtered"
EXPECTED_SCHEMA_VERSION = "0.1"
EXPECTED_FRAME_ID = "map"
EXPECTED_GRID_SIZE_M = 0.4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class StaticOccupancyMetadata:
    schema_version: str
    representation: str
    frame_id: str
    grid_size_xy_m: float
    column_count: int
    semantic_labels: bool
    scene_id: str
    scene_uuid: str
    revision: int
    npz_file_sha256: str
    content_sha256: str
    source_service: str
    storage: str


@dataclass(frozen=True)
class StaticOccupancyColumns:
    metadata: StaticOccupancyMetadata
    grid_i: np.ndarray
    grid_j: np.ndarray
    min_z: np.ndarray
    max_z: np.ndarray
    point_count: np.ndarray
    flags: np.ndarray
    source_dir: Path

    def __len__(self) -> int:
        return self.metadata.column_count

    @property
    def xy_centers_map(self) -> np.ndarray:
        scale = self.metadata.grid_size_xy_m
        return np.column_stack(((self.grid_i.astype(np.float64) + 0.5) * scale,
                                (self.grid_j.astype(np.float64) + 0.5) * scale))

    @property
    def xyz_interval_centers_map(self) -> np.ndarray:
        xy = self.xy_centers_map
        z = (self.min_z.astype(np.float64) + self.max_z.astype(np.float64)) * 0.5
        return np.column_stack((xy, z))

    def cell_bounds_xy_map(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        scale = self.metadata.grid_size_xy_m
        x0 = self.grid_i.astype(np.float64) * scale
        y0 = self.grid_j.astype(np.float64) * scale
        return x0, x0 + scale, y0, y0 + scale


def _required(metadata: dict[str, Any], name: str) -> Any:
    if name not in metadata:
        raise ValueError(f"static occupancy metadata missing {name!r}")
    return metadata[name]


def _parse_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value in {"True", "False"}:
        return value == "True"
    raise ValueError(f"{name} must be bool or 'True'/'False', got {value!r}")


def read_static_occupancy(clip_or_static_dir: str | Path, *, verify_hash: bool = True) -> StaticOccupancyColumns:
    path = Path(clip_or_static_dir).expanduser().resolve()
    static_dir = path if path.name == "static_occupancy" else path / "static_occupancy"
    metadata_path = static_dir / "metadata.json"
    npz_path = static_dir / "columns.npz"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)

    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("static occupancy metadata must be a JSON object")
    metadata = StaticOccupancyMetadata(
        schema_version=str(_required(raw, "schema_version")),
        representation=str(_required(raw, "representation")),
        frame_id=str(_required(raw, "frame_id")),
        grid_size_xy_m=float(_required(raw, "grid_size_xy_m")),
        column_count=int(_required(raw, "column_count")),
        semantic_labels=_parse_bool(_required(raw, "semantic_labels"), "semantic_labels"),
        scene_id=str(_required(raw, "scene_id")),
        scene_uuid=str(_required(raw, "scene_uuid")),
        revision=int(_required(raw, "revision")),
        npz_file_sha256=str(_required(raw, "npz_file_sha256")),
        content_sha256=str(_required(raw, "content_sha256")),
        source_service=str(_required(raw, "source_service")),
        storage=str(_required(raw, "storage")),
    )
    if metadata.schema_version != EXPECTED_SCHEMA_VERSION:
        raise ValueError(f"unsupported static occupancy schema: {metadata.schema_version}")
    if metadata.representation != EXPECTED_REPRESENTATION:
        raise ValueError(f"unsupported representation: {metadata.representation}")
    if metadata.frame_id != EXPECTED_FRAME_ID:
        raise ValueError(f"unsupported frame_id: {metadata.frame_id}")
    if not np.isclose(metadata.grid_size_xy_m, EXPECTED_GRID_SIZE_M, rtol=0.0, atol=1e-6):
        raise ValueError(f"unexpected grid size: {metadata.grid_size_xy_m}")
    if metadata.semantic_labels:
        raise ValueError("schema 0.1 static occupancy must be semantic-label free")
    if verify_hash:
        actual_hash = sha256_file(npz_path)
        if actual_hash != metadata.npz_file_sha256:
            raise ValueError(f"columns.npz SHA256 mismatch: metadata={metadata.npz_file_sha256}, actual={actual_hash}")

    with np.load(npz_path, allow_pickle=False) as archive:
        actual_names = set(archive.files)
        expected_names = set(EXPECTED_ARRAYS)
        if actual_names != expected_names:
            raise ValueError(f"unexpected arrays: expected={sorted(expected_names)}, actual={sorted(actual_names)}")
        arrays = {name: np.asarray(archive[name]) for name in EXPECTED_ARRAYS}

    for name, expected_dtype in EXPECTED_ARRAYS.items():
        array = arrays[name]
        if array.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional, got {array.shape}")
        if array.dtype != expected_dtype:
            raise ValueError(f"{name} dtype must be {expected_dtype}, got {array.dtype}")
        if len(array) != metadata.column_count:
            raise ValueError(f"{name} length {len(array)} != column_count {metadata.column_count}")
        array.setflags(write=False)

    if metadata.column_count <= 0:
        raise ValueError("static occupancy must contain at least one column")
    if not np.isfinite(arrays["min_z"]).all() or not np.isfinite(arrays["max_z"]).all():
        raise ValueError("min_z and max_z must be finite")
    if np.any(arrays["max_z"] <= arrays["min_z"]):
        raise ValueError("each column must satisfy max_z > min_z")
    if np.any(arrays["point_count"] < 1):
        raise ValueError("point_count must be positive")

    return StaticOccupancyColumns(metadata=metadata, source_dir=static_dir, **arrays)
