from __future__ import annotations
import argparse
import json
from pathlib import Path
from torch.utils.data import DataLoader
from .dataset import AlpaSimOccupancyDataset, collate_alpasim_samples
from .manifests import ManifestIndex
from .occstudio_bridge import to_occstudio_batch


def main():
    parser = argparse.ArgumentParser(description="Audit the delivered Step 7 DataLoader")
    parser.add_argument("--sample-id")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    index = ManifestIndex()
    sample_id = args.sample_id or index.sample_ids[0]
    dataset = AlpaSimOccupancyDataset([sample_id])
    batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0,
                                 collate_fn=collate_alpasim_samples)))
    bridge = to_occstudio_batch(batch)
    result = {
        "status": "pass",
        "sample_id": sample_id,
        "dataset_samples": len(index),
        "tensor_shapes": {
            key: list(batch[key].shape) for key in (
                "imgs", "rots", "trans", "intrins", "post_rots",
                "post_trans", "bda", "gt_occupancy", "mask_camera", "mask_lidar"
            )
        },
        "occstudio_img_inputs": [list(value.shape) for value in bridge["img_inputs"]],
        "img_metas": bridge["img_metas"],
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"audit_output={path}")


if __name__ == "__main__":
    main()
