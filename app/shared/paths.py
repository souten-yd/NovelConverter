"""Centralised data-directory resolution.

Priority:
1. DATA_DIR env var  (set to /workspace/data in Docker)
2. <repo_root>/data  (local dev fallback)
"""
from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent  # NovelConverter/

def get_data_dir() -> Path:
    env = os.environ.get("DATA_DIR", "")
    if env:
        return Path(env)
    return _REPO_ROOT / "data"
