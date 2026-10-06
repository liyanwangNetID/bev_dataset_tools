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
    metadata = [sample.pop("metadata") for sample in samples]
    raw_images = [sample.pop("raw_images_uint8") for sample in samples]
    images = [sample.pop("images_uint8") for sample in samples]
    valid_masks = [sample.pop("image_valid_masks") for sample in samples]
    batch = default_collate(samples)
    batch["metadata"] = metadata
    batch["raw_images_uint8"] = raw_images
    batch["images_uint8"] = images
    batch["image_valid_masks"] = valid_masks
    return batch
