from step6.finalize import classify_edge_cases


def test_edge_cases_pass_with_complete_consistent_diagnosis():
    diagnosis = {
        "record_count": 2,
        "zero_empty_count": 0,
        "zero_nonempty_count": 1,
        "full_visibility_count": 1,
        "records": [
            {
                "sample_id": "rear",
                "known_voxels": 5,
                "mask_camera_true": 0,
                "recomputed_visible": 0,
                "known_quadrants": {
                    "front_left": 0, "front_right": 0, "rear_left": 2, "rear_right": 3
                },
                "camera_projection_counts": {"front_wide": {"valid_projection": 0}},
            },
            {
                "sample_id": "full-small",
                "known_voxels": 12,
                "mask_camera_true": 12,
                "recomputed_visible": 12,
                "known_quadrants": {
                    "front_left": 0, "front_right": 12, "rear_left": 0, "rear_right": 0
                },
                "camera_projection_counts": {"cross_right": {"valid_projection": 12}},
            },
        ],
    }
    result = classify_edge_cases(diagnosis)
    assert result["status"] == "pass"
    assert result["zero_nonempty_count"] == 1
    assert result["full_visibility_count"] == 1


def test_edge_case_fails_for_recomputed_visibility_mismatch():
    diagnosis = {
        "record_count": 1,
        "zero_empty_count": 0,
        "zero_nonempty_count": 1,
        "full_visibility_count": 0,
        "records": [
            {
                "sample_id": "bad",
                "known_voxels": 5,
                "mask_camera_true": 0,
                "recomputed_visible": 1,
                "known_quadrants": {
                    "front_left": 5, "front_right": 0, "rear_left": 0, "rear_right": 0
                },
                "camera_projection_counts": {"front_wide": {"valid_projection": 0}},
            }
        ],
    }
    result = classify_edge_cases(diagnosis)
    assert result["status"] == "fail"
    assert result["violations"] == ["bad: recomputed visibility mismatch"]
