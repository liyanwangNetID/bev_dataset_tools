from __future__ import annotations
import json
from pathlib import Path
from .constants import KEYFRAME_MANIFEST, LABEL_MANIFEST, SOURCE_MANIFEST


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: {exc}") from exc


class ManifestIndex:
    """Join Step 4/6 formal products by sample_id."""
    def __init__(self, keyframes=KEYFRAME_MANIFEST, labels=LABEL_MANIFEST,
                 sources=SOURCE_MANIFEST):
        self.paths = tuple(map(Path, (keyframes, labels, sources)))
        for path in self.paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        self._records = {}
        for name, path in zip(("keyframe", "label", "source"), self.paths):
            for row in _read_jsonl(path):
                sample_id = row.get("sample_id")
                if not sample_id:
                    raise KeyError(f"{path}: record has no sample_id")
                slot = self._records.setdefault(sample_id, {})
                if name in slot:
                    raise ValueError(f"duplicate {name} record for {sample_id}")
                slot[name] = row

    def __len__(self):
        return len(self._records)

    @property
    def sample_ids(self):
        return tuple(sorted(self._records))

    def get(self, sample_id):
        try:
            joined = self._records[sample_id]
        except KeyError as exc:
            raise KeyError(f"unknown sample_id: {sample_id}") from exc
        missing = {"keyframe", "label", "source"} - set(joined)
        if missing:
            raise KeyError(f"{sample_id} missing manifest records: {sorted(missing)}")
        return joined
