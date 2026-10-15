"""Pytest path setup for both package-style and legacy top-level imports."""
from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPOSITORY_ROOT.parent

for path in (PROJECT_ROOT, REPOSITORY_ROOT):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)
