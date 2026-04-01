"""Orchestrator – main FastAPI application."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.shared.database import init_db
from app.shared.logger import get_logger
from app.orchestrator.routers import projects, processing, segments, voice_mapping, render, ui

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
app.include_router(render.router)
app.include_router(ui.router)  # must be last (catch-all pages)


@app.on_event("startup")
def on_startup():
    init_db()
    logger.info("NovelConverter Orchestrator started")


@app.get("/health")
def health():
    return {"status": "ok", "service": "orchestrator"}
