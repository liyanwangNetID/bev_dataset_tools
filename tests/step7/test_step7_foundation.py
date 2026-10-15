import numpy as np
import pytest
from bev_dataset_tools.step7.calibration import canonical_hash
from bev_dataset_tools.step7.dataset import AlpaSimOccupancyDataset
from bev_dataset_tools.step7.manifests import ManifestIndex


def test_canonical_hash_is_order_independent():
    assert canonical_hash({"b": 2, "a": 1}) == canonical_hash({"a": 1, "b": 2})


def test_manifest_index_rejects_unknown_sample(tmp_path):
    paths = []
    for name in ("keyframes.jsonl", "labels.jsonl", "sources.jsonl"):
        path = tmp_path / name
        path.write_text("", encoding="utf-8")
        paths.append(path)
    index = ManifestIndex(*paths)
    with pytest.raises(KeyError, match="unknown sample_id"):
        index.get("does_not_exist")


def test_dataset_uses_requested_id_order():
    class Loader:
        index = type("Index", (), {"sample_ids": ("a", "b")})()
        def load(self, sample_id):
            return {"sample_id": sample_id}
    dataset = AlpaSimOccupancyDataset(["b", "a"], loader=Loader())
    assert dataset[0]["sample_id"] == "b"
    assert dataset[1]["sample_id"] == "a"
