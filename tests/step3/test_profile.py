from step3.profile import build_grid, build_summary, nearest_timestamp, select_candidate_pairs


def test_grid_and_candidate_pair_selection():
    assert build_grid(100, 1_000_000_100) == [100, 500_000_100, 1_000_000_100]
    stamps = [100, 100_000_100, 500_000_100, 600_000_100, 1_000_000_100]
    assert nearest_timestamp(stamps, 500_000_100, 5_000_000) == 500_000_100
    assert select_candidate_pairs(stamps) == [(100, 500_000_100), (500_000_100, 1_000_000_100)]


def test_summary_uses_candidate_pairs():
    records = [{"temporal_eligible": True, "grid_temporal_pair_count": 0, "valid_temporal_pair_count": 3, "synchronized_grid_ratio": 0.2, "ineligibility_reasons": [], "resolution_pairs": ["854x480->1920x1080"], "resolution_matches_calibration": False}]
    summary = build_summary(records)
    assert summary["total_grid_temporal_pairs"] == 0
    assert summary["total_valid_temporal_pairs"] == 3
    assert summary["temporal_eligible_clips"] == 1
