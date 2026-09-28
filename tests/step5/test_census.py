from step5.census import canonical_json, split_calibration


def test_canonical_json_is_order_independent():
    assert canonical_json({"b": 2, "a": 1}) == canonical_json({"a": 1, "b": 2})


def test_split_calibration_separates_intrinsics_and_extrinsics():
    data = {
        "available_camera": {
            "logical_id": "camera",
            "intrinsics": {"resolution_w": 1920},
            "rig_to_camera": {"vec": {"x": 1}},
        }
    }
    intrinsics, extrinsics = split_calibration(data)
    assert "intrinsics" in intrinsics
    assert "rig_to_camera" not in intrinsics
    assert "rig_to_camera" in extrinsics
    assert "intrinsics" not in extrinsics
