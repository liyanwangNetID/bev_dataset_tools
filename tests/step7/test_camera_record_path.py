from pathlib import Path
import pytest

from bev_dataset_tools.step7.loader import _camera_path_value, _resolve


def test_camera_path_value_supports_formal_dictionary():
    record = {
        "image_path": "test_clip_001/cameras/front_wide/000001.jpg",
        "timestamp_ns": 123,
        "frame_index": 1,
    }
    assert _camera_path_value(record, "front_wide") == record["image_path"]


def test_camera_path_value_supports_direct_string():
    assert _camera_path_value("clip/camera/1.jpg") == "clip/camera/1.jpg"


def test_camera_path_value_finds_unique_image_suffix():
    record = {"uri": "clip/camera/1.png", "timestamp_ns": 123}
    assert _camera_path_value(record) == "clip/camera/1.png"


def test_camera_path_value_rejects_metadata_only_record():
    with pytest.raises(KeyError, match="no unique image path"):
        _camera_path_value({"timestamp_ns": 123}, "cross_left")


def test_resolve_uses_raw_root_for_relative_formal_path(tmp_path):
    result = _resolve(tmp_path, {"image_path": "clip/cam/1.jpg"}, camera="front_wide")
    assert result == tmp_path / "clip/cam/1.jpg"
