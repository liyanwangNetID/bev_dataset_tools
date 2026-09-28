"""Shared path configuration for the BEV AlpaSim dataset pipeline.

The raw Clip tree is treated as read-only. Generated artifacts are written
under the dedicated outcome root. Both roots can be overridden through
environment variables without modifying source code.
"""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_ALPASIM_DATA_ROOT = Path("/home/lab/data_from_alpasim")
DEFAULT_OUTCOME_ROOT = Path("/home/lab/bev_alpasim_dataset_tools/outcome")


def _path_from_environment(name: str, default: Path) -> Path:
    raw_value = os.environ.get(name)
    if raw_value is None or not raw_value.strip():
        return default
    return Path(raw_value).expanduser()


ALPASIM_DATA_ROOT = _path_from_environment(
    "BEV_ALPASIM_DATA_ROOT",
    DEFAULT_ALPASIM_DATA_ROOT,
)
OUTCOME_ROOT = _path_from_environment(
    "BEV_DATASET_OUTCOME_ROOT",
    DEFAULT_OUTCOME_ROOT,
)

SCHEMA_ROOT = OUTCOME_ROOT / "schemas"
MANIFEST_ROOT = OUTCOME_ROOT / "manifests"
REPORT_ROOT = OUTCOME_ROOT / "reports"
REVIEW_ROOT = OUTCOME_ROOT / "review"
DATASET_ROOT = OUTCOME_ROOT / "datasets"


def resolved_roots() -> dict[str, Path]:
    """Return resolved configured roots for logging and diagnostics."""
    return {
        "alpasim_data_root": ALPASIM_DATA_ROOT.expanduser().resolve(),
        "outcome_root": OUTCOME_ROOT.expanduser().resolve(),
    }
