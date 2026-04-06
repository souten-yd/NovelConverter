"""Qwen3-TTS model download, verification, and path management.

Downloads the 4 required Qwen3-TTS models from HuggingFace Hub on first run,
verifies integrity, and provides path resolution for all TTS workers.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, Optional

from app.shared.logger import get_logger
from app.shared.paths import get_cache_root, get_models_root

logger = get_logger("model_manager")

# ── Model registry ───────────────────────────────────────────────────────────

QWEN3_TTS_MODELS: Dict[str, dict] = {
    "tokenizer": {
        "repo_id": "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        "local_dir": "Qwen3-TTS-Tokenizer-12Hz",
        "required_files": ["config.json"],
        "description": "Shared speech tokenizer (12 Hz)",
    },
    "custom": {
        "repo_id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "local_dir": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "required_files": ["config.json"],
        "description": "CustomVoice 1.7B (named speaker mode)",
    },
    "design": {
        "repo_id": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "local_dir": "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "required_files": ["config.json"],
        "description": "VoiceDesign 1.7B (natural description mode)",
    },
    "base": {
        "repo_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "local_dir": "Qwen3-TTS-12Hz-1.7B-Base",
        "required_files": ["config.json"],
        "description": "Base 1.7B (voice clone mode)",
    },
}


def _models_root() -> Path:
    return get_models_root() / "qwen3_tts"


# ── Path helpers ─────────────────────────────────────────────────────────────

def get_model_path(model_key: str) -> Path:
    """Return local directory path for a Qwen3-TTS model."""
    info = QWEN3_TTS_MODELS[model_key]
    return _models_root() / info["local_dir"]


def get_tokenizer_path() -> Path:
    return get_model_path("tokenizer")


# ── Verification ─────────────────────────────────────────────────────────────

def _has_weight_files(model_dir: Path) -> bool:
    """Check if model directory contains weight files (safetensors or bin)."""
    for ext in ("*.safetensors", "*.bin", "*.pt", "*.onnx"):
        if list(model_dir.glob(ext)):
            return True
    return False


def is_model_complete(model_key: str) -> bool:
    """Check whether a model is fully downloaded and has required files."""
    info = QWEN3_TTS_MODELS[model_key]
    model_dir = get_model_path(model_key)

    if not model_dir.is_dir():
        return False

    # Check required config files
    for req in info["required_files"]:
        if not (model_dir / req).is_file():
            return False

    # Tokenizer only needs config + processor files, not weights
    if model_key == "tokenizer":
        return True

    # Model directories need weight files
    return _has_weight_files(model_dir)


def get_models_status() -> Dict[str, dict]:
    """Return download/completeness status for all models (for UI display)."""
    result = {}
    for key, info in QWEN3_TTS_MODELS.items():
        model_dir = get_model_path(key)
        exists = model_dir.is_dir()
        complete = is_model_complete(key) if exists else False
        result[key] = {
            "repo_id": info["repo_id"],
            "local_path": str(model_dir),
            "exists": exists,
            "complete": complete,
            "status": "ok" if complete else ("incomplete" if exists else "missing"),
            "description": info["description"],
        }
    return result


# ── Download ─────────────────────────────────────────────────────────────────

def ensure_model(model_key: str) -> Path:
    """Download a model if missing or incomplete. Returns local path."""
    info = QWEN3_TTS_MODELS[model_key]
    model_dir = get_model_path(model_key)

    if is_model_complete(model_key):
        logger.info(f"[model_manager] {model_key}: already present, skipping download ({model_dir})")
        return model_dir

    state = "incomplete" if model_dir.exists() else "missing"
    logger.info(f"[model_manager] {model_key}: {state}, {'re-downloading' if state == 'incomplete' else 'downloading'}")

    # Remove incomplete directory
    if model_dir.exists():
        logger.warning(f"[model_manager] {model_key}: incomplete, re-downloading ({model_dir})")
        shutil.rmtree(model_dir, ignore_errors=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise RuntimeError(
            "huggingface_hub is required for model download. "
            "Install with: pip install huggingface_hub"
        )

    repo_id = info["repo_id"]
    logger.info(f"[model_manager] Downloading {repo_id} to {model_dir} ...")

    _models_root().mkdir(parents=True, exist_ok=True)

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(model_dir),
            cache_dir=str(get_cache_root() / "huggingface"),
        )
    except Exception as e:
        logger.error(f"[model_manager] Failed to download {repo_id}: {e}")
        # Clean up partial download
        if model_dir.exists():
            shutil.rmtree(model_dir, ignore_errors=True)
        raise

    if not is_model_complete(model_key):
        raise RuntimeError(
            f"Download of {repo_id} completed but verification failed. "
            f"Required files not found in {model_dir}"
        )

    logger.info(f"[model_manager] {model_key} download complete: {model_dir}")
    return model_dir


def ensure_all_models() -> Dict[str, bool]:
    """Download all required models. Returns dict of model_key → success."""
    results = {}
    for key in QWEN3_TTS_MODELS:
        try:
            ensure_model(key)
            results[key] = True
        except Exception as e:
            logger.error(f"[model_manager] Failed to ensure {key}: {e}")
            results[key] = False
    return results


def ensure_models_for_mode(mode: str) -> Dict[str, bool]:
    """Download only the models needed for a specific TTS mode.

    mode: 'custom', 'design', or 'base' (clone)
    Always includes the tokenizer.
    """
    needed = ["tokenizer", mode]
    results = {}
    for key in needed:
        try:
            ensure_model(key)
            results[key] = True
        except Exception as e:
            logger.error(f"[model_manager] Failed to ensure {key}: {e}")
            results[key] = False
    return results
