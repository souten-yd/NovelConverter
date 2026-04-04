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
from app.orchestrator.routers import projects, processing, segments, voice_mapping, render, ui, llm
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
app.include_router(voice_mapping.router)
app.include_router(tts_router)
app.include_router(render.router)
app.include_router(llm.router)
app.include_router(ui.router)  # must be last (catch-all pages)


@app.on_event("startup")
def on_startup():
    init_db()
    logger.info("NovelConverter Orchestrator started")


@app.get("/health")
def health():
    return {"status": "ok", "service": "orchestrator"}


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
