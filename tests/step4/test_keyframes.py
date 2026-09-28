from step4.keyframes import validate_records


def test_validate_scene_start_and_temporal_records():
    base = {"sample_schema_version":"0.1","generator_version":"0.1.0","scene_id":"scene","camera_order":["cross_left","front_wide","cross_right","front_tele"]}
    records = [
        {**base,"sample_id":"scene_100","current_timestamp_ns":100,"history_timestamp_ns":100,"history_delta_ns":0,"is_scene_start":True,"history_is_repeated_current":True},
        {**base,"sample_id":"scene_500000100","current_timestamp_ns":500000100,"history_timestamp_ns":100,"history_delta_ns":500000000,"is_scene_start":False,"history_is_repeated_current":False},
    ]
    validate_records(records)

def test_scene_start_selects_first_pose_covered_common_timestamp(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import step4.keyframes as module

    class FakeReader:
        clip_id = "test_clip_001"

        def __init__(self, clip_dir):
            self.clip_dir = clip_dir

        def get_ego_pose3d_at(self, stamp_ns, maximum_interpolation_span_ns):
            return None if stamp_ns == 100 else SimpleNamespace(exact=True)

    monkeypatch.setattr(module, "DrivingClipReader", FakeReader)
    monkeypatch.setattr(module, "common_timestamps", lambda reader: [100, 200])
    monkeypatch.setattr(module, "select_candidate_pairs", lambda stamps: [])

    captured = []

    def fake_make_record(reader, dataset_root, history_ns, current_ns, scene_start):
        captured.append((history_ns, current_ns, scene_start))
        return ({
            "sample_id": "test_clip_001_200",
            "scene_id": "test_clip_001",
            "current_timestamp_ns": 200,
        }, [])

    monkeypatch.setattr(module, "make_record", fake_make_record)
    records, rejections = module.inspect_clip(tmp_path / "test_clip_001", tmp_path)
    assert captured == [(200, 200, True)]
    assert len(records) == 1
    assert rejections == []
