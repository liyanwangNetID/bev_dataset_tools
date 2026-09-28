from pathlib import Path

from step5.finalize import build_assignments


def test_build_assignments_is_camera_ordered():
    census = [
        {
            "clip_id": "test_clip_001",
            "camera_name": "front_tele",
            "rectification_asset_id": "tele",
            "intrinsic_hash": "i4",
            "extrinsic_hash": "e4",
            "full_calibration_hash": "f4",
            "valid_pixel_ratio": 1.0,
        },
        {
            "clip_id": "test_clip_001",
            "camera_name": "cross_left",
            "rectification_asset_id": "left",
            "intrinsic_hash": "i1",
            "extrinsic_hash": "e1",
            "full_calibration_hash": "f1",
            "valid_pixel_ratio": 0.86,
        },
        {
            "clip_id": "test_clip_001",
            "camera_name": "cross_right",
            "rectification_asset_id": "right",
            "intrinsic_hash": "i3",
            "extrinsic_hash": "e3",
            "full_calibration_hash": "f3",
            "valid_pixel_ratio": 0.86,
        },
        {
            "clip_id": "test_clip_001",
            "camera_name": "front_wide",
            "rectification_asset_id": "wide",
            "intrinsic_hash": "i2",
            "extrinsic_hash": "e2",
            "full_calibration_hash": "f2",
            "valid_pixel_ratio": 0.85,
        },
    ]
    assignments, lookup = build_assignments(census)
    assert [item["camera_name"] for item in assignments] == [
        "cross_left",
        "front_wide",
        "cross_right",
        "front_tele",
    ]
    assert lookup[("test_clip_001", "front_wide")] == "wide"
