from __future__ import annotations
import argparse
from torch.utils.data import DataLoader
from .dataset import AlpaSimOccupancyDataset, collate_alpasim_samples
from .loader import Step7SampleLoader
from .visualization import save_validation


def main():
    parser = argparse.ArgumentParser(description="Load and visualize one formal AlpaSim occupancy sample")
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    loader = Step7SampleLoader()
    dataset = AlpaSimOccupancyDataset([args.sample_id], loader=loader)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=collate_alpasim_samples)
    batch = next(iter(dataloader))
    sample = loader.load(args.sample_id)
    save_validation(sample, args.output_dir)
    print(f"sample_id={args.sample_id}")
    for key in ("imgs", "rots", "trans", "intrins", "gt_occupancy", "mask_camera"):
        print(f"batch.{key}: shape={tuple(batch[key].shape)} dtype={batch[key].dtype}")
    print(f"validation_output={args.output_dir}")


if __name__ == "__main__":
    main()
