from pathlib import Path
import numpy as np

from step2.static_occupancy_reader import StaticOccupancyColumns, StaticOccupancyMetadata
from step6.contract import GRID, UNKNOWN_ID
from step6.voxelize import voxelize_sample


def _static_columns():
    metadata = StaticOccupancyMetadata(
        schema_version="0.1", representation="single_interval_2p5d_static_occupancy_road_filtered",
        frame_id="map", grid_size_xy_m=0.4, column_count=1, semantic_labels=False,
        scene_id="scene", scene_uuid="uuid", revision=1, npz_file_sha256="hash",
        content_sha256="content", source_service="service", storage="npz_compressed",
    )
    return StaticOccupancyColumns(
        metadata=metadata, grid_i=np.array([0], dtype=np.int32),
        grid_j=np.array([0], dtype=np.int32), min_z=np.array([-0.2], dtype=np.float32),
        max_z=np.array([0.6], dtype=np.float32), point_count=np.array([1], dtype=np.uint32),
        flags=np.array([1], dtype=np.uint16), source_dir=Path("/tmp/static_occupancy"),
    )


def test_voxelize_sample_includes_static_id_zero_before_visibility():
    source = {"sources": {"actors": []}}
    keyframe = {"current_ego2global": np.eye(4).tolist()}
    semantics, stats = voxelize_sample(source, keyframe, {"lanes": []}, _static_columns())
    assert semantics[100, 100, 2] == 0
    assert semantics[100, 100, 3] == 0
    assert stats["static_source_columns"] == 1
    assert stats["static_voxel_writes"] == 2
    assert stats["unknown_voxels"] == semantics.size - 2


def test_voxelize_sample_remains_callable_without_static_for_unit_compatibility():
    source = {"sources": {"actors": []}}
    keyframe = {"current_ego2global": np.eye(4).tolist()}
    semantics, stats = voxelize_sample(source, keyframe, {"lanes": []})
    assert np.all(semantics == UNKNOWN_ID)
    assert stats["static_source_columns"] == 0
