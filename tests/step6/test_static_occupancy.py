from __future__ import annotations
from pathlib import Path
import numpy as np

from step2.static_occupancy_reader import StaticOccupancyColumns, StaticOccupancyMetadata
from step6.contract import GRID, UNKNOWN_ID, OBSERVED_FREE_ID
from step6.static_occupancy import STATIC_OCCUPANCY_ID, voxelize_static_occupancy


def _columns(grid_i=(0,), grid_j=(0,), min_z=(-0.2,), max_z=(0.6,)):
    count = len(grid_i)
    metadata = StaticOccupancyMetadata(
        schema_version="0.1",
        representation="single_interval_2p5d_static_occupancy_road_filtered",
        frame_id="map",
        grid_size_xy_m=0.4,
        column_count=count,
        semantic_labels=False,
        scene_id="scene",
        scene_uuid="uuid",
        revision=1,
        npz_file_sha256="hash",
        content_sha256="content",
        source_service="service",
        storage="npz_compressed",
    )
    return StaticOccupancyColumns(
        metadata=metadata,
        grid_i=np.asarray(grid_i, dtype=np.int32),
        grid_j=np.asarray(grid_j, dtype=np.int32),
        min_z=np.asarray(min_z, dtype=np.float32),
        max_z=np.asarray(max_z, dtype=np.float32),
        point_count=np.ones(count, dtype=np.uint32),
        flags=np.ones(count, dtype=np.uint16),
        source_dir=Path("/tmp/static_occupancy"),
    )


def test_identity_transform_writes_expected_static_voxels():
    semantics = np.full(GRID.shape, UNKNOWN_ID, dtype=np.uint8)
    stats = voxelize_static_occupancy(semantics, _columns(), np.eye(4))
    occupied = np.argwhere(semantics == STATIC_OCCUPANCY_ID)
    assert stats.written_static_voxels == 2
    assert occupied.tolist() == [[100, 100, 2], [100, 100, 3]]


def test_ego_translation_selects_map_column():
    semantics = np.full(GRID.shape, UNKNOWN_ID, dtype=np.uint8)
    ego2map = np.eye(4)
    ego2map[0, 3] = 4.0
    columns = _columns(grid_i=(10,), grid_j=(0,))
    voxelize_static_occupancy(semantics, columns, ego2map)
    occupied = np.argwhere(semantics == STATIC_OCCUPANCY_ID)
    assert occupied[:, 0].tolist() == [100, 100]


def test_specific_semantics_are_preserved_and_free_can_be_replaced():
    semantics = np.full(GRID.shape, UNKNOWN_ID, dtype=np.uint8)
    semantics[100, 100, 2] = 11
    semantics[100, 100, 3] = OBSERVED_FREE_ID
    stats = voxelize_static_occupancy(semantics, _columns(), np.eye(4))
    assert semantics[100, 100, 2] == 11
    assert semantics[100, 100, 3] == STATIC_OCCUPANCY_ID
    assert stats.preserved_semantic_voxels == 1
    assert stats.replaced_observed_free_voxels == 1


def test_rejects_wrong_grid_shape():
    semantics = np.full((2, 2, 2), UNKNOWN_ID, dtype=np.uint8)
    try:
        voxelize_static_occupancy(semantics, _columns(), np.eye(4))
    except ValueError as error:
        assert "shape" in str(error)
    else:
        raise AssertionError("expected ValueError")
