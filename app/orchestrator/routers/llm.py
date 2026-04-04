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
    batch_size: int = 512
    threads: int = -1
    flash_attn: bool = False


class ModelSettingsRequest(BaseModel):
    model_filename: str
    n_gpu_layers: int = -1
    ctx_size: int = 4096
    batch_size: int = 512
    threads: int = -1
    flash_attn: bool = False
    auto_unload_seconds: int = 0  # 0 = disabled
    is_main_model: bool = False


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/status")
def get_status():
    return llm_manager.get_server_status_dict()


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
            batch_size=body.batch_size,
            threads=body.threads,
            flash_attn=body.flash_attn,
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
        "stderr": state.stderr,
        "load_params": state.load_params,
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


# ── Per-model settings ────────────────────────────────────────────────────────

@router.post("/settings")
def save_model_settings(body: ModelSettingsRequest):
    """Persist per-model load settings and auto-unload config."""
    updates = {
        "n_gpu_layers": body.n_gpu_layers,
        "ctx_size": body.ctx_size,
        "batch_size": body.batch_size,
        "threads": body.threads,
        "flash_attn": body.flash_attn,
        "auto_unload_seconds": body.auto_unload_seconds,
        "is_main_model": body.is_main_model,
    }
    merged = llm_manager.update_model_setting(body.model_filename, updates)

    # If auto_unload changed, the running watcher will pick it up on next cycle
    logger.info(f"Saved settings for model {body.model_filename!r}: {updates}")
    return {"saved": True, "model_filename": body.model_filename, "settings": merged}


@router.get("/settings/{model_filename:path}")
def get_model_settings(model_filename: str):
    """Return stored settings for a model (merged with defaults)."""
    return llm_manager.get_model_setting(model_filename)


@router.get("/main_model")
def get_main_model():
    """Return the filename of the model marked as 'main', if any."""
    main = llm_manager.get_main_model()
    return {"main_model": main}
