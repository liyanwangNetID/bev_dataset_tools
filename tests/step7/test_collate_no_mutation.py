import numpy as np
import torch
from bev_dataset_tools.step7.dataset import collate_alpasim_samples


def _sample(include_visual=True):
    sample = {
        "sample_id": "sample_1",
        "clip_id": "clip_1",
        "camera_order": ("cross_left", "front_wide", "cross_right", "front_tele"),
        "imgs": torch.zeros(2, 4, 3, 4, 4),
        "rots": torch.zeros(2, 4, 3, 3),
        "trans": torch.zeros(2, 4, 3),
        "intrins": torch.zeros(2, 4, 3, 3),
        "post_rots": torch.zeros(2, 4, 3, 3),
        "post_trans": torch.zeros(2, 4, 3),
        "bda": torch.eye(3),
        "gt_occupancy": torch.zeros(2, 2, 2, dtype=torch.long),
        "gt_occupancy_ori": torch.zeros(2, 2, 2, dtype=torch.long),
        "mask_camera": torch.ones(2, 2, 2, dtype=torch.bool),
        "mask_lidar": torch.zeros(2, 2, 2, dtype=torch.bool),
        "metadata": {"keyframe": {"sample_id": "sample_1"}},
    }
    if include_visual:
        sample.update({
            "raw_images_uint8": np.zeros((2, 4, 4, 4, 3), dtype=np.uint8),
            "images_uint8": np.zeros((2, 4, 4, 4, 3), dtype=np.uint8),
            "image_valid_masks": np.ones((2, 4, 4, 4), dtype=bool),
        })
    return sample


def test_collate_does_not_mutate_input_and_can_repeat():
    sample = _sample(include_visual=True)
    original_keys = set(sample)
    first = collate_alpasim_samples([sample])
    second = collate_alpasim_samples([sample])
    assert set(sample) == original_keys
    assert first["raw_images_uint8"][0].shape == (2, 4, 4, 4, 3)
    assert second["imgs"].shape == (1, 2, 4, 3, 4, 4)


def test_collate_accepts_sample_without_visual_arrays():
    sample = _sample(include_visual=False)
    batch = collate_alpasim_samples([sample])
    assert batch["imgs"].shape == (1, 2, 4, 3, 4, 4)
    assert "raw_images_uint8" not in batch
    assert "images_uint8" not in batch
    assert "image_valid_masks" not in batch
