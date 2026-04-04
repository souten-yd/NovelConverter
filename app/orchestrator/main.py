"""Orchestrator – main FastAPI application."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import requests

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.shared.database import init_db
from app.shared.logger import get_logger
from app.orchestrator.routers import projects, processing, segments, voice_mapping, render, ui, llm, characters, ocr_viewer
from app.orchestrator.routers.voice_mapping import tts_router

logger = get_logger("orchestrator")

app = FastAPI(title="NovelConverter Orchestrator", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Routers
app.include_router(projects.router)
app.include_router(processing.router)
app.include_router(segments.router)
app.include_router(characters.router)
app.include_router(voice_mapping.router)
app.include_router(tts_router)
app.include_router(render.router)
app.include_router(llm.router)
app.include_router(ocr_viewer.router)
app.include_router(ui.router)  # must be last (catch-all pages)


@app.on_event("startup")
def on_startup():
    init_db()
    logger.info("NovelConverter Orchestrator started")
    _log_ocr_engine_status()


def _log_ocr_engine_status() -> None:
    """Log availability of all OCR engines at startup."""
    try:
        from app.orchestrator.services.ocr.factory import list_engines
        engines = list_engines()
        for eng in engines:
            if eng["available"]:
                logger.info(f"OCR engine [{eng['id']}] {eng['name']}: OK")
            else:
                logger.warning(
                    f"OCR engine [{eng['id']}] {eng['name']}: UNAVAILABLE – "
                    f"{eng['missing_deps']}"
                )
        # Extra detail for PaddleOCR version
        try:
            from app.orchestrator.services.ocr.paddleocr_engine import _get_paddleocr_version
            logger.info(f"PaddleOCR installed version: {_get_paddleocr_version()}")
        except Exception:
            pass
        # Extra detail for NDLOCR-Lite
        try:
            from app.orchestrator.services.ocr.ndlocr_lite_engine import get_ndlocr_status
            ndl_status = get_ndlocr_status()
            logger.info(
                f"NDLOCR-Lite status – installed={ndl_status['installed']} "
                f"cli_ok={ndl_status['cli_functional']} "
                f"models={ndl_status['model_files_present']} "
                f"model_path={ndl_status['model_path']}"
            )
            if ndl_status["error_message"]:
                logger.warning(f"NDLOCR-Lite: {ndl_status['error_message']}")
        except Exception:
            pass
    except Exception as exc:
        logger.warning(f"OCR engine status check failed at startup: {exc}")


@app.get("/health")
def health():
    return {"status": "ok", "service": "orchestrator"}


@app.get("/api/ocr/engines")
def list_ocr_engines():
    """List available OCR engines with their status."""
    from app.orchestrator.services.ocr.factory import list_engines
    return list_engines()


@app.get("/api/ocr/engines/ndlocr_lite/status")
def ndlocr_lite_status():
    """Detailed runtime status for the NDLOCR-Lite engine.

    Returns installed / cli_functional / model_files_present / runtime_ready
    and the model_path so operators can confirm what is (or isn't) set up.
    """
    from app.orchestrator.services.ocr.ndlocr_lite_engine import get_ndlocr_status
    return get_ndlocr_status()


@app.get("/api/ocr/engines/paddleocr/status")
def paddleocr_status():
    """Detailed runtime status for the PaddleOCR engine."""
    from app.orchestrator.services.ocr.paddleocr_engine import (
        _get_paddleocr_version,
        PaddleOCREngine,
    )
    version = _get_paddleocr_version()
    init_error = PaddleOCREngine._init_error
    return {
        "version": version,
        "instance_ready": PaddleOCREngine._ocr_instance is not None,
        "init_error": init_error,
        "runtime_ready": init_error is None,
    }


@app.get("/api/health/dependencies")
def check_dependencies():
    """Check availability of all optional dependencies."""
    from app.orchestrator.services.ocr.factory import list_engines

    ocr_engines = list_engines()

    # Check archive dependencies
    archive_deps = {}
    try:
        import rarfile  # noqa: F401
        archive_deps["rarfile"] = {"available": True}
    except ImportError:
        archive_deps["rarfile"] = {"available": False, "install": "pip install rarfile"}

    import shutil
    for cmd in ["unrar", "7z", "bsdtar"]:
        archive_deps[cmd] = {"available": shutil.which(cmd) is not None}

    return {
        "ocr_engines": ocr_engines,
        "archive": archive_deps,
    }


@app.get("/api/status")
def full_status() -> Dict[str, Any]:
    """Aggregate health check for orchestrator + all TTS workers.
    Used by CI smoke tests and the Docker HEALTHCHECK.
    """
    worker_urls = {
        "tts_base":   os.environ.get("TTS_BASE_URL",   "http://localhost:8001"),
        "tts_custom": os.environ.get("TTS_CUSTOM_URL", "http://localhost:8002"),
        "tts_design": os.environ.get("TTS_DESIGN_URL", "http://localhost:8003"),
    }

    workers: Dict[str, Any] = {}
    all_ok = True

    for name, url in worker_urls.items():
        try:
            r = requests.get(f"{url}/health", timeout=5)
            if r.status_code == 200:
                workers[name] = {"status": "ok", **r.json()}
            else:
                workers[name] = {"status": "error", "http_status": r.status_code}
                all_ok = False
        except Exception as e:
            workers[name] = {"status": "unreachable", "error": str(e)}
            all_ok = False

    return {
        "orchestrator": {"status": "ok"},
        "workers": workers,
        "all_healthy": all_ok,
    }


@app.get("/api/system/models")
def system_models_status() -> Dict[str, Any]:
    """Return status of all models: Qwen3-TTS + LLM."""
    result: Dict[str, Any] = {}

    # Qwen3-TTS models
    try:
        from app.shared.model_manager import get_models_status
        result["qwen3_tts"] = get_models_status()
    except Exception as e:
        result["qwen3_tts"] = {"error": str(e)}

    # LLM model
    try:
        from app.shared.llm_downloader import get_llm_status
        result["llm"] = get_llm_status()
    except Exception as e:
        result["llm"] = {"error": str(e)}

    # LLM runtime (llama-server)
    try:
        from app.orchestrator.services.llm_manager import get_refcount_status
        result["llm_runtime"] = get_refcount_status()
    except Exception as e:
        result["llm_runtime"] = {"error": str(e)}

    # TTS worker statuses
    worker_urls = {
        "tts_base":   os.environ.get("TTS_BASE_URL",   "http://localhost:8001"),
        "tts_custom": os.environ.get("TTS_CUSTOM_URL", "http://localhost:8002"),
        "tts_design": os.environ.get("TTS_DESIGN_URL", "http://localhost:8003"),
    }
    workers = {}
    for name, url in worker_urls.items():
        try:
            r = requests.get(f"{url}/model_status", timeout=5)
            if r.status_code == 200:
                workers[name] = r.json()
            else:
                workers[name] = {"error": f"HTTP {r.status_code}"}
        except Exception as e:
            workers[name] = {"error": str(e)}
    result["tts_workers"] = workers

    return result
