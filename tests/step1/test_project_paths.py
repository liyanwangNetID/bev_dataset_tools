from __future__ import annotations

import importlib
from pathlib import Path


def test_default_paths(monkeypatch):
    monkeypatch.delenv("BEV_ALPASIM_DATA_ROOT", raising=False)
    monkeypatch.delenv("BEV_DATASET_OUTCOME_ROOT", raising=False)

    import project_paths

    module = importlib.reload(project_paths)
    assert module.ALPASIM_DATA_ROOT == Path("/home/lab/data_from_alpasim")
    assert module.OUTCOME_ROOT == Path(
        "/home/lab/bev_alpasim_dataset_tools/outcome"
    )
    assert module.SCHEMA_ROOT == module.OUTCOME_ROOT / "schemas"
    assert module.MANIFEST_ROOT == module.OUTCOME_ROOT / "manifests"
    assert module.REPORT_ROOT == module.OUTCOME_ROOT / "reports"


def test_environment_overrides(monkeypatch, tmp_path):
    raw_root = tmp_path / "raw"
    outcome_root = tmp_path / "outcome"
    monkeypatch.setenv("BEV_ALPASIM_DATA_ROOT", str(raw_root))
    monkeypatch.setenv("BEV_DATASET_OUTCOME_ROOT", str(outcome_root))

    import project_paths

    module = importlib.reload(project_paths)
    assert module.ALPASIM_DATA_ROOT == raw_root
    assert module.OUTCOME_ROOT == outcome_root
    assert module.DATASET_ROOT == outcome_root / "datasets"
