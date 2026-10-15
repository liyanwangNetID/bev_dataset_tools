from step5.trial import evenly_spaced_scene_ids, select_trial_records


def _records(scene_count):
    rows=[]
    for index in range(1, scene_count + 1):
        scene=f"test_clip_{index:03d}"
        rows.append({"scene_id":scene,"sample_id":f"{scene}_start","is_scene_start":True})
        rows.append({"scene_id":scene,"sample_id":f"{scene}_middle","is_scene_start":False})
    return rows


def test_evenly_spaced_selection_uses_active_manifest_scale():
    scene_ids=tuple(f"test_clip_{index:03d}" for index in range(1,81))
    assert evenly_spaced_scene_ids(scene_ids,5) == (
        "test_clip_001","test_clip_020","test_clip_040","test_clip_060","test_clip_080"
    )


def test_small_manifest_uses_all_available_scenes():
    scene_ids=("a","b","c")
    assert evenly_spaced_scene_ids(scene_ids,5) == scene_ids


def test_default_trial_records_are_derived_from_manifest():
    selected=select_trial_records(_records(80))
    assert [row["scene_id"] for row in selected] == [
        "test_clip_001","test_clip_020","test_clip_040","test_clip_060","test_clip_080"
    ]
    assert all(row["sample_id"].endswith("_middle") for row in selected)


def test_explicit_missing_scene_still_fails():
    try:
        select_trial_records(_records(3),("test_clip_999",))
    except ValueError as error:
        assert "scene not present" in str(error)
    else:
        raise AssertionError("expected ValueError")
