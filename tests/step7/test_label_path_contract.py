from pathlib import Path

from bev_dataset_tools.step7.loader import _first, _resolve


def test_formal_label_manifest_uses_labels_path(tmp_path):
    record = {
        "labels_path": "occupancy/labels_v0.1/clip/sample/labels.npz",
        "sample_id": "clip_sample",
    }
    value = _first(record, "labels_path", "label_path", "occ_path", "path")
    assert value == record["labels_path"]
    assert _resolve(tmp_path, value) == tmp_path / value


def test_labels_path_has_priority_over_legacy_aliases():
    record = {
        "labels_path": "formal.npz",
        "label_path": "legacy.npz",
        "occ_path": "older.npz",
    }
    assert _first(record, "labels_path", "label_path", "occ_path", "path") == "formal.npz"
