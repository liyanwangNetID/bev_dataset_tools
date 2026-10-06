# Step 7 AlpaSim Occupancy DataLoader

## Purpose

This package turns one formal AlpaSim occupancy `sample_id` into model-ready
camera inputs, calibration tensors, occupancy labels, masks, metadata, and
visual review artifacts. Training policy and model optimization are outside
this package.

## Sample identity

A sample ID has the form:

```text
{scene_id}_{current_timestamp_ns}
```

Example:

```text
test_clip_001_9300112661000
```

The example identifies one keyframe sample inside scene `test_clip_001`.

## Frozen contracts

- Time order: `current`, then `history`.
- Camera order: `cross_left`, `front_wide`, `cross_right`, `front_tele`.
- Rectified image size: `960 x 540`.
- Occupancy shape: `200 x 200 x 16`.
- Free label: `17`.
- Ignore/unknown label: `255`.
- Camera-only supervision uses `mask_camera`.
- `mask_lidar` may be all false and is not fabricated.
- Scene starts repeat current as history.
- `camera_to_ego` is obtained by inverting formal `rig_to_camera`.

## Environment variables

```bash
export ALPASIM_RAW_DATA_ROOT=/home/lab/data_from_alpasim
export BEV_DATASET_PROJECT_ROOT=/home/lab/bev_alpasim_dataset_tools
export BEV_DATASET_OUTCOME_ROOT=/home/lab/bev_alpasim_dataset_tools/outcome
```

Defaults match the development machine. Delivery users should set these
variables when their paths differ.

## Validate one sample

```bash
cd /home/lab/bev_alpasim_dataset_tools
python3 -m bev_dataset_tools.step7.validate_dataloader_sample \
  --sample-id test_clip_001_9300112661000 \
  --output-dir /tmp/step7_review
```

## List samples in a scene

```bash
python3 -m bev_dataset_tools.step7.list_samples \
  --scene-id test_clip_001 \
  --limit 30
```

## Use the DataLoader

```python
from torch.utils.data import DataLoader
from bev_dataset_tools.step7.dataset import (
    AlpaSimOccupancyDataset,
    collate_alpasim_samples,
)

sample_ids = ["test_clip_001_9300112661000"]
dataset = AlpaSimOccupancyDataset(sample_ids)
dataloader = DataLoader(
    dataset,
    batch_size=1,
    num_workers=0,
    collate_fn=collate_alpasim_samples,
)
batch = next(iter(dataloader))
```

## Output tensors

```text
imgs          [B, 2, 4, 3, 540, 960] float32
rots          [B, 2, 4, 3, 3]        float32
trans         [B, 2, 4, 3]           float32
intrins       [B, 2, 4, 3, 3]        float32
post_rots     [B, 2, 4, 3, 3]        float32
post_trans    [B, 2, 4, 3]           float32
bda           [B, 3, 3]               float32
gt_occupancy  [B, 200, 200, 16]       int64
mask_camera   [B, 200, 200, 16]       bool
mask_lidar    [B, 200, 200, 16]       bool
```

## OccStudio bridge

```python
from bev_dataset_tools.step7.occstudio_bridge import to_occstudio_batch
model_batch = to_occstudio_batch(batch)
```

The bridge returns `img_inputs`, `gt_occupancy`, masks, and `img_metas`. It is
a format adapter only. Model training remains the responsibility of the model
owner.

## Tests and delivery audit

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/step7 -q
python3 -m bev_dataset_tools.step7.audit_delivery \
  --sample-id test_clip_001_9300112661000 \
  --output /tmp/step7_delivery_audit.json
```

## Validation artifacts

The validator writes raw and rectified camera mosaics, semantic BEV,
`mask_camera` BEV, all 16 height slices, a JSON summary, and compressed tensor
arrays. Always inspect these artifacts when moving the dataset to a new
machine or replacing calibration products.
