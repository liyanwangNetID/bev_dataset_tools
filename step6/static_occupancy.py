"""Voxelize map-frame 2.5D static columns into the current-ego occupancy grid."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from step2.static_occupancy_reader import StaticOccupancyColumns
from step6.contract import GRID, OBSERVED_FREE_ID, UNKNOWN_ID

STATIC_OCCUPANCY_ID = 0


@dataclass(frozen=True)
class StaticVoxelizationStats:
    source_columns: int
    target_voxel_centers: int
    matched_column_voxels: int
    in_height_interval_voxels: int
    written_static_voxels: int
    preserved_semantic_voxels: int
    replaced_observed_free_voxels: int


def _target_centers_current_ego() -> np.ndarray:
    indices = np.indices(GRID.shape, dtype=np.int16).reshape(3, -1).T
    origin = np.array([GRID.x_min, GRID.y_min, GRID.z_min], dtype=np.float64)
    return origin + (indices.astype(np.float64) + 0.5) * GRID.voxel_size


def _packed_keys(grid_i: np.ndarray, grid_j: np.ndarray) -> np.ndarray:
    # Lossless pair packing for signed int32 values.
    return (grid_i.astype(np.int64) << 32) | (grid_j.astype(np.int64) & 0xFFFFFFFF)


def voxelize_static_occupancy(
    semantics: np.ndarray,
    columns: StaticOccupancyColumns,
    ego2map: np.ndarray,
) -> StaticVoxelizationStats:
    """Write unlabeled static occupancy as ID 0 without overwriting IDs 1-16.

    Target voxel centers are transformed from current ego to the source map frame.
    Their map XY location selects a source 0.4 m column, and map Z must lie within
    that column's [min_z, max_z] interval. ID 255 and any pre-existing ID 17 may
    be replaced by ID 0. Specific occupied semantics 1-16 are preserved.
    """
    if semantics.shape != GRID.shape:
        raise ValueError(f"semantics shape must be {GRID.shape}, got {semantics.shape}")
    if semantics.dtype != np.uint8:
        raise ValueError(f"semantics dtype must be uint8, got {semantics.dtype}")
    transform = np.asarray(ego2map, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"ego2map must be 4x4, got {transform.shape}")
    if not np.isfinite(transform).all():
        raise ValueError("ego2map must be finite")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
        raise ValueError("ego2map must be a homogeneous rigid transform")

    centers_ego = _target_centers_current_ego()
    homogeneous = np.column_stack((centers_ego, np.ones(len(centers_ego))))
    centers_map = (homogeneous @ transform.T)[:, :3]
    scale = columns.metadata.grid_size_xy_m
    query_i = np.floor(centers_map[:, 0] / scale).astype(np.int64)
    query_j = np.floor(centers_map[:, 1] / scale).astype(np.int64)
    query_keys = _packed_keys(query_i, query_j)

    source_keys = _packed_keys(columns.grid_i, columns.grid_j)
    order = np.argsort(source_keys, kind="stable")
    sorted_keys = source_keys[order]
    positions = np.searchsorted(sorted_keys, query_keys)
    valid_position = positions < len(sorted_keys)
    matched = np.zeros(len(query_keys), dtype=bool)
    matched[valid_position] = sorted_keys[positions[valid_position]] == query_keys[valid_position]

    source_index = np.zeros(len(query_keys), dtype=np.int64)
    source_index[matched] = order[positions[matched]]
    in_interval = np.zeros(len(query_keys), dtype=bool)
    z = centers_map[:, 2]
    selected = source_index[matched]
    in_interval[matched] = (
        (z[matched] >= columns.min_z[selected].astype(np.float64))
        & (z[matched] <= columns.max_z[selected].astype(np.float64))
    )

    flat = semantics.reshape(-1)
    previous = flat.copy()
    writable = in_interval & ((previous == UNKNOWN_ID) | (previous == OBSERVED_FREE_ID))
    flat[writable] = np.uint8(STATIC_OCCUPANCY_ID)
    preserved = in_interval & ~writable

    return StaticVoxelizationStats(
        source_columns=len(columns),
        target_voxel_centers=len(flat),
        matched_column_voxels=int(matched.sum()),
        in_height_interval_voxels=int(in_interval.sum()),
        written_static_voxels=int(writable.sum()),
        preserved_semantic_voxels=int(preserved.sum()),
        replaced_observed_free_voxels=int(np.count_nonzero(writable & (previous == OBSERVED_FREE_ID))),
    )
