#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from app.shared.llm_downloader import ensure_llm_model, is_llm_ready
from app.shared.logger import get_logger
from app.shared.model_manager import QWEN3_TTS_MODELS, ensure_model, is_model_complete

logger = get_logger("startup.prepare_assets")

DATA_DIR = Path(os.environ.get("DATA_DIR", "/workspace/data"))
STATE_FILE = DATA_DIR / "state" / "startup_status.json"
OCR_PYTHON = os.environ.get("OCR_PYTHON", "/opt/venvs/ocr/bin/python")


def _write_status(task: str, progress: int, detail: str, status: str) -> None:
    payload = {
        "phase": "prepare_assets",
        "current_task": task,
        "progress": int(progress),
        "detail": detail,
        "status": status,
    }
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _check_ndlocr_assets() -> None:
    ndl_root = Path("/opt/ndlocr-lite")
    ocr_py = ndl_root / "src/ocr.py"
    model_dir = ndl_root / "src/model"
    config_dir = ndl_root / "src/config"
    model_count = len(list(model_dir.glob("*.onnx"))) if model_dir.exists() else 0

    missing = []
    if not ocr_py.is_file():
        missing.append(str(ocr_py))
    if model_count != 4:
        missing.append(f"{model_dir}/*.onnx count={model_count}")
    if not (config_dir / "ndl.yaml").is_file():
        missing.append(str(config_dir / "ndl.yaml"))
    if not (config_dir / "NDLmoji.yaml").is_file():
        missing.append(str(config_dir / "NDLmoji.yaml"))

    if missing:
        raise RuntimeError(f"NDLOCR-Lite assets missing: {', '.join(missing)}")
    logger.info("[prepare_assets] NDLOCR-Lite assets: ok")


def _prepare_paddle_models() -> None:
    # Remove known-incomplete PaddleX model caches so a clean auto-download can happen.
    models_base = Path("/root/.paddlex/official_models")
    for model_name in ("PP-LCNet_x1_0_doc_ori", "PP-ShiTuV2_det", "PP-ShiTuV2_rec"):
        model_dir = models_base / model_name
        if model_dir.is_dir() and not (model_dir / "inference.yml").is_file():
            logger.warning(
                "[prepare_assets] Incomplete PaddleX model cache detected, removing: %s",
                model_dir,
            )
            import shutil

            shutil.rmtree(model_dir, ignore_errors=True)

    if not Path(OCR_PYTHON).exists():
        logger.warning("[prepare_assets] OCR python missing at %s; skip Paddle warmup", OCR_PYTHON)
        return

    cmd = [
        OCR_PYTHON,
        "-c",
        (
            "from app.orchestrator.services.ocr.paddleocr_engine import PaddleOCREngine;"
            "PaddleOCREngine.configure(device='gpu:0', use_layout=False);"
            "PaddleOCREngine._init_ocr('japan');"
            "print('PaddleOCR warmup ok')"
        ),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"PaddleOCR prepare failed: {(proc.stderr or proc.stdout).strip()}")
    logger.info("[prepare_assets] PaddleOCR models: ok")


def _prepare_qwen_tts_models() -> None:
    for key in QWEN3_TTS_MODELS:
        if is_model_complete(key):
            logger.info("[prepare_assets] qwen3-tts %s: ok", key)
            continue
        logger.info("[prepare_assets] qwen3-tts %s: missing, downloading", key)
        ensure_model(key)


def _prepare_llm_model() -> None:
    if is_llm_ready():
        logger.info("[prepare_assets] llm: ok")
        return
    logger.info("[prepare_assets] llm: missing, downloading")
    ensure_llm_model()


def main() -> int:
    tasks = [
        ("check_ndlocr_assets", 15, _check_ndlocr_assets),
        ("prepare_paddle_models", 35, _prepare_paddle_models),
        ("prepare_qwen3tts", 75, _prepare_qwen_tts_models),
        ("download_llm", 95, _prepare_llm_model),
    ]

    try:
        for task_name, progress, fn in tasks:
            _write_status(task_name, progress, f"{task_name} running", "running")
            fn()
        _write_status("done", 100, "all assets ready", "ok")
        logger.info("[prepare_assets] all assets ready")
        return 0
    except Exception as exc:
        logger.exception("[prepare_assets] failed: %s", exc)
        _write_status(task_name, progress, str(exc), "fail")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
