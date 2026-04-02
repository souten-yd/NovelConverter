"""LLM management API – search, download, load/unload GGUF models for llama.cpp."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.orchestrator.services import llm_manager
from app.shared.logger import get_logger

logger = get_logger("router.llm")
router = APIRouter(prefix="/api/llm", tags=["llm"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    limit: int = 20


class FilesRequest(BaseModel):
    repo_id: str


class DownloadRequest(BaseModel):
    repo_id: str
    filename: str


class LoadRequest(BaseModel):
    model_filename: str
    n_gpu_layers: int = -1
    ctx_size: int = 4096


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/status")
def get_status():
    state = llm_manager.get_server_status()
    return {
        "status": state.status,
        "model_filename": state.model_filename,
        "model_path": state.model_path,
        "port": state.port,
        "pid": state.pid,
        "error": state.error,
    }


@router.get("/models")
def list_models():
    return llm_manager.list_local_models()


@router.post("/search")
def search_models(body: SearchRequest):
    results = llm_manager.search_huggingface(body.query, body.limit)
    return results


@router.post("/files")
def get_model_files(body: FilesRequest):
    files = llm_manager.get_model_files(body.repo_id)
    return files


@router.post("/download")
def download_model(body: DownloadRequest):
    task_key = llm_manager.start_download(body.repo_id, body.filename)
    return {"task_key": task_key, "status": "downloading"}


@router.get("/download/status")
def download_status(task_key: str):
    return llm_manager.get_download_status(task_key)


@router.post("/load")
def load_model(body: LoadRequest):
    try:
        state = llm_manager.load_model(
            body.model_filename,
            n_gpu_layers=body.n_gpu_layers,
            ctx_size=body.ctx_size,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {
        "status": state.status,
        "model_filename": state.model_filename,
        "port": state.port,
        "error": state.error,
    }


@router.post("/unload")
def unload_model():
    llm_manager.unload_model()
    return {"status": "stopped"}


@router.delete("/models/{model_filename:path}")
def delete_model(model_filename: str):
    try:
        llm_manager.delete_model(model_filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"deleted": True, "filename": model_filename}
