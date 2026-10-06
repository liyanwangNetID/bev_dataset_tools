from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from .loader import Step7SampleLoader


class AlpaSimOccupancyDataset(Dataset):
    def __init__(self, sample_ids=None, loader=None):
        self.loader = loader or Step7SampleLoader()
        self.sample_ids = tuple(sample_ids or self.loader.index.sample_ids)
    def __len__(self):
        return len(self.sample_ids)
    def __getitem__(self, index):
        return self.loader.load(self.sample_ids[index])


def collate_alpasim_samples(samples):
    """Collate without mutating Dataset samples.

    Visualization-only arrays are optional so the delivery audit works with
    both the basic loader and the upgraded visual-validation loader.
    """
    metadata = [sample["metadata"] for sample in samples]
    optional_keys = (
        "raw_images_uint8",
        "images_uint8",
        "image_valid_masks",
    )
    optional_values = {
        key: [sample.get(key) for sample in samples]
        for key in optional_keys
    }
    excluded = {"metadata", *optional_keys}
    tensor_samples = [
        {key: value for key, value in sample.items() if key not in excluded}
        for sample in samples
    ]
    batch = default_collate(tensor_samples)
    batch["metadata"] = metadata
    for key, values in optional_values.items():
        if all(value is not None for value in values):
            batch[key] = values
    return batch
