from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from step2.static_occupancy_reader import read_static_occupancy


def _write_fixture(root: Path, *, bad_hash=False, bad_interval=False) -> Path:
    static_dir = root / "test_clip_001" / "static_occupancy"
    static_dir.mkdir(parents=True)
    arrays = {
        "grid_i": np.array([-1, 2], dtype=np.int32),
        "grid_j": np.array([3, -4], dtype=np.int32),
        "min_z": np.array([-0.2, 1.0], dtype=np.float32),
        "max_z": np.array([0.6, 0.5 if bad_interval else 2.2], dtype=np.float32),
        "point_count": np.array([3, 9], dtype=np.uint32),
        "flags": np.array([1, 3], dtype=np.uint16),
    }
    npz = static_dir / "columns.npz"
    np.savez_compressed(npz, **arrays)
    digest = hashlib.sha256(npz.read_bytes()).hexdigest()
    metadata = {
        "schema_version": "0.1",
        "representation": "single_interval_2p5d_static_occupancy_road_filtered",
        "frame_id": "map",
        "grid_size_xy_m": "0.4000000059604645",
        "column_count": "2",
        "semantic_labels": "False",
        "scene_id": "scene",
        "scene_uuid": "uuid",
        "revision": "1",
        "npz_file_sha256": "bad" if bad_hash else digest,
        "content_sha256": "content",
        "source_service": "/alpasim/static_occupancy/get",
        "storage": "npz_compressed",
    }
    (static_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return static_dir.parent


def test_reads_schema_and_map_cell_centers(tmp_path):
    clip = _write_fixture(tmp_path)
    columns = read_static_occupancy(clip)
    assert len(columns) == 2
    np.testing.assert_allclose(columns.xy_centers_map, [[-0.2, 1.4], [1.0, -1.4]], atol=1e-6)
    assert columns.metadata.frame_id == "map"
    assert not columns.metadata.semantic_labels
    assert not columns.grid_i.flags.writeable


def test_rejects_hash_mismatch(tmp_path):
    clip = _write_fixture(tmp_path, bad_hash=True)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        read_static_occupancy(clip)


def test_rejects_inverted_height_interval(tmp_path):
    clip = _write_fixture(tmp_path, bad_interval=True)
    with pytest.raises(ValueError, match="max_z > min_z"):
        read_static_occupancy(clip)


def test_real_clip_contract_if_available():
    clip = Path("/home/lab/data_from_alpasim/test_clip_001")
    if not clip.is_dir():
        return
    columns = read_static_occupancy(clip)
    assert len(columns) == 100290
    assert columns.metadata.representation.endswith("road_filtered")
    assert np.isclose(columns.metadata.grid_size_xy_m, 0.4, atol=1e-6)
    assert set(np.unique(columns.flags)).issubset({1, 2, 3, 4, 5, 6})
