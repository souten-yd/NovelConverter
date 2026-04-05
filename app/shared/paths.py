"""Cross-platform path resolution helpers."""
from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def get_repo_root() -> Path:
    return _REPO_ROOT


def get_data_dir() -> Path:
    env = os.environ.get("DATA_DIR", "")
    data = Path(env) if env else (_REPO_ROOT / "data")
    data.mkdir(parents=True, exist_ok=True)
    return data


def get_models_root() -> Path:
    root = Path(os.environ.get("MODELS_ROOT", str(get_data_dir() / "models")))
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_cache_root() -> Path:
    root = Path(os.environ.get("CACHE_ROOT", str(_REPO_ROOT / "cache")))
    root.mkdir(parents=True, exist_ok=True)
    return root
