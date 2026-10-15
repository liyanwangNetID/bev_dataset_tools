import json
from pathlib import Path

from bev_dataset_tools.step7.calibration import CalibrationResolver, canonical_hash


def test_asset_match_compares_raw_candidate_hash(tmp_path):
    raw_root = tmp_path / "raw"
    rect_root = tmp_path / "rect"
    calibration_dir = raw_root / "clip_1" / "calibration"
    assets_dir = rect_root / "intrinsics_assets"
    calibration_dir.mkdir(parents=True)

    source_a = {"logical_id": "cam_id", "intrinsics": {"f": 100.0}}
    source_b = {"logical_id": "cam_id", "intrinsics": {"f": 200.0}}
    raw = {
        "calibration": source_b,
        "camera_to_ego": [
            [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]
        ],
    }

    for camera in ("cross_left", "front_wide", "cross_right", "front_tele"):
        (calibration_dir / f"{camera}.json").write_text(json.dumps(raw))
        for suffix, source in (("a", source_a), ("b", source_b)):
            directory = assets_dir / f"{camera}_{suffix}"
            directory.mkdir(parents=True)
            metadata = {
                "asset_id": f"{camera}_{suffix}",
                "camera_name": camera,
                "intrinsic_hash": canonical_hash(source),
                "source_intrinsics": source,
                "K_rect": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "valid_pixel_ratio": 1.0,
            }
            (directory / "metadata.json").write_text(json.dumps(metadata))
            (directory / "maps.npz").touch()

    resolver = CalibrationResolver(raw_root=raw_root, rectification_root=rect_root)
    resolved = resolver.resolve("clip_1", {camera: "cam_id" for camera in (
        "cross_left", "front_wide", "cross_right", "front_tele")})
    assert {value["asset_id"] for value in resolved.values()} == {
        "cross_left_b", "front_wide_b", "cross_right_b", "front_tele_b"
    }
