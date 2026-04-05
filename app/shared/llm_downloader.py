"""Gemma 4 GGUF auto-download for LLM inference.

Downloads the main LLM model on first run and integrates with llm_manager's
models directory so it appears in list_local_models().
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from app.shared.logger import get_logger
from app.shared.paths import get_data_dir

logger = get_logger("llm_downloader")

LLM_REPO = "unsloth/gemma-4-E4B-it-GGUF"
LLM_FILENAME = "gemma-4-E4B-it-Q4_K_M.gguf"

# Minimum expected file size (bytes) – Gemma 4 Q4_K_M is ~2.5 GB+
_MIN_FILE_SIZE = 1_000_000_000  # 1 GB sanity check


def _llm_models_dir() -> Path:
    """LLM models directory, shared with llm_manager."""
    d = get_data_dir() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_llm_model_path() -> Path:
    return _llm_models_dir() / LLM_FILENAME


def is_llm_ready() -> bool:
    """Check if the LLM GGUF file exists and passes size check."""
    p = get_llm_model_path()
    if not p.is_file():
        return False
    return p.stat().st_size >= _MIN_FILE_SIZE


def ensure_llm_model() -> Path:
    """Download the Gemma 4 GGUF if missing or corrupt. Returns local path."""
    dest = get_llm_model_path()

    if is_llm_ready():
        logger.info(f"[llm_downloader] LLM model: already present, skipping download ({dest})")
        return dest

    logger.info(
        f"[llm_downloader] LLM model: "
        f"{'incomplete, re-downloading' if dest.exists() else 'missing, downloading'} ({dest})"
    )

    # Remove corrupt/partial file
    if dest.exists():
        logger.warning(f"[llm_downloader] LLM file incomplete, removing: {dest}")
        dest.unlink()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError(
            "huggingface_hub is required for LLM download. "
            "Install with: pip install huggingface_hub"
        )

    logger.info(f"[llm_downloader] Downloading {LLM_REPO}/{LLM_FILENAME} ...")

    models_dir = _llm_models_dir()
    try:
        local_path = hf_hub_download(
            repo_id=LLM_REPO,
            filename=LLM_FILENAME,
            local_dir=str(models_dir),
            local_dir_use_symlinks=False,
        )
        # hf_hub_download may put file in a subdirectory; move to top level
        src = Path(local_path)
        if src != dest and src.exists():
            src.rename(dest)

    except Exception as e:
        logger.error(f"[llm_downloader] Download failed: {e}")
        if dest.exists():
            dest.unlink()
        raise

    if not is_llm_ready():
        raise RuntimeError(
            f"LLM download completed but file is missing or too small: {dest}"
        )

    logger.info(f"[llm_downloader] LLM model ready: {dest} ({dest.stat().st_size / 1e9:.1f} GB)")
    return dest


def get_llm_status() -> dict:
    """Return LLM model status for UI display."""
    dest = get_llm_model_path()
    exists = dest.is_file()
    size = dest.stat().st_size if exists else 0
    ready = is_llm_ready()
    return {
        "filename": LLM_FILENAME,
        "repo_id": LLM_REPO,
        "path": str(dest),
        "exists": exists,
        "size_bytes": size,
        "size_gb": round(size / 1e9, 2) if size else 0,
        "ready": ready,
        "status": "ok" if ready else ("corrupt" if exists else "missing"),
    }
