import torch
from bev_dataset_tools.step7.config import Step7Paths
from bev_dataset_tools.step7.occstudio_bridge import to_occstudio_batch


def test_path_config_uses_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("ALPASIM_RAW_DATA_ROOT", str(tmp_path / "raw"))
    monkeypatch.setenv("BEV_DATASET_OUTCOME_ROOT", str(tmp_path / "outcome"))
    paths = Step7Paths.from_environment()
    assert paths.raw_data_root == (tmp_path / "raw").resolve()
    assert paths.outcome_root == (tmp_path / "outcome").resolve()


def test_occstudio_bridge_contract():
    batch = {
        "imgs": torch.zeros(1, 2, 4, 3, 8, 8),
        "rots": torch.zeros(1, 2, 4, 3, 3),
        "trans": torch.zeros(1, 2, 4, 3),
        "intrins": torch.zeros(1, 2, 4, 3, 3),
        "post_rots": torch.zeros(1, 2, 4, 3, 3),
        "post_trans": torch.zeros(1, 2, 4, 3),
        "bda": torch.zeros(1, 3, 3),
        "gt_occupancy": torch.zeros(1, 200, 200, 16, dtype=torch.long),
        "mask_camera": torch.zeros(1, 200, 200, 16, dtype=torch.bool),
        "mask_lidar": torch.zeros(1, 200, 200, 16, dtype=torch.bool),
        "metadata": [{"keyframe": {
            "sample_id": "scene_1", "scene_id": "scene", "is_scene_start": True,
            "history_is_repeated_current": True, "history_delta_ns": 0,
            "camera_order": ["cross_left", "front_wide", "cross_right", "front_tele"],
        }}],
    }
    result = to_occstudio_batch(batch)
    assert len(result["img_inputs"]) == 7
    assert result["img_metas"][0]["start_of_sequence"] is True
    assert result["img_metas"][0]["camera_order"][1] == "front_wide"
