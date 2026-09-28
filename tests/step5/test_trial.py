from step5.trial import CAMERA_NAMES, TARGET_HEIGHT, TARGET_WIDTH, select_trial_records, target_config


def test_target_configs():
    for camera in CAMERA_NAMES:
        config = target_config(camera)
        assert config.resolution_hw == (TARGET_HEIGHT, TARGET_WIDTH)
        assert not config.radial
        assert not config.tangential
        assert not config.thin_prism
    assert target_config("front_wide").focal_length == (480.0, 480.0)
    assert target_config("front_tele").focal_length == (1925.175, 1925.175)


def test_select_trial_records_prefers_temporal_middle():
    records = [
        {"scene_id":"scene","is_scene_start":True,"sample_id":"start"},
        {"scene_id":"scene","is_scene_start":False,"sample_id":"a"},
        {"scene_id":"scene","is_scene_start":False,"sample_id":"b"},
        {"scene_id":"scene","is_scene_start":False,"sample_id":"c"},
    ]
    assert select_trial_records(records, ("scene",))[0]["sample_id"] == "b"
