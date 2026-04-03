"""LLM server manager – launch/stop llama.cpp server, manage GGUF models."""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import requests

from app.shared.logger import get_logger
from app.shared.paths import get_data_dir

logger = get_logger("llm_manager")

LLAMA_SERVER_PORT = int(os.environ.get("LLAMA_SERVER_PORT", "8080"))
MODELS_DIR_NAME = "models"

_STARTUP_TIMEOUT = 60  # seconds to wait for llama-server to become ready
_HEALTH_INTERVAL = 2   # seconds between health-check polls


# ── State ─────────────────────────────────────────────────────────────────────

@dataclass
class LlmServerState:
    status: str = "stopped"   # stopped / loading / running / error
    model_filename: str = ""
    model_path: str = ""
    port: int = LLAMA_SERVER_PORT
    pid: Optional[int] = None
    error: str = ""


_state = LlmServerState()
_proc: Optional[subprocess.Popen] = None
_lock = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _models_dir() -> Path:
    d = get_data_dir() / MODELS_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def _api_base() -> str:
    return f"http://127.0.0.1:{_state.port}"


def _is_server_healthy() -> bool:
    try:
        r = requests.get(f"{_api_base()}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _get_llama_server_bin() -> str:
    """Read LLAMA_SERVER_BIN env var at call time so changes take effect without restart."""
    return os.environ.get("LLAMA_SERVER_BIN", "llama-server")


def _llama_server_candidates() -> List[str]:
    """Return binary candidates in discovery order."""
    configured = os.environ.get("LLAMA_SERVER_BIN")
    if configured:
        return [configured]
    return [
        "llama-server",
        "llama-server.exe",
        "./llama-server",
        "./build/bin/llama-server",
        "./llama.cpp/build/bin/llama-server",
        "/opt/llama-cpp/bin/llama-server",
    ]


def _resolve_llama_server_bin() -> Optional[str]:
    """Resolve llama-server path from env or common local build locations."""
    for candidate in _llama_server_candidates():
        expanded = Path(candidate).expanduser()
        if expanded.is_file():
            return str(expanded)
        found = shutil.which(candidate)
        if found:
            return found
    return None


def check_binary_available() -> bool:
    """Return True if the llama-server binary can be found."""
    return _resolve_llama_server_bin() is not None


def _update_env_url(url: str) -> None:
    """Patch the process env so speaker_segmenter picks up the new URL."""
    os.environ["LLM_API_URL"] = url


# ── Public API ────────────────────────────────────────────────────────────────

def get_server_status() -> LlmServerState:
    global _state, _proc
    with _lock:
        if _state.status == "running" and _proc is not None:
            if _proc.poll() is not None:
                # process died
                _state.status = "error"
                _state.error = f"llama-server exited with code {_proc.returncode}"
                _proc = None
        return LlmServerState(**_state.__dict__)


def load_model(
    model_filename: str,
    n_gpu_layers: int = -1,
    ctx_size: int = 4096,
) -> LlmServerState:
    global _state, _proc

    with _lock:
        if _state.status in ("loading", "running"):
            raise RuntimeError("A model is already loaded. Unload it first.")

        model_path = _models_dir() / model_filename
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_filename}")

        _state = LlmServerState(
            status="loading",
            model_filename=model_filename,
            model_path=str(model_path),
            port=LLAMA_SERVER_PORT,
        )

    llama_bin = _resolve_llama_server_bin()
    if not llama_bin:
        configured_bin = _get_llama_server_bin()
        with _lock:
            _state.status = "error"
            _state.error = (
                f"llama-server binary not found at '{configured_bin}'. "
                "Set LLAMA_SERVER_BIN env var or place llama-server under ./build/bin."
            )
        logger.error(_state.error)
        return LlmServerState(**_state.__dict__)

    cmd = [
        llama_bin,
        "--model", str(model_path),
        "--port", str(LLAMA_SERVER_PORT),
        "--n-gpu-layers", str(n_gpu_layers),
        "--ctx-size", str(ctx_size),
        "--host", "127.0.0.1",
    ]
    logger.info(f"Launching llama-server: {' '.join(cmd)}")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except FileNotFoundError:
        configured_bin = _get_llama_server_bin()
        with _lock:
            _state.status = "error"
            _state.error = (
                f"llama-server binary not found at '{configured_bin}'. "
                "Set LLAMA_SERVER_BIN env var or place llama-server under ./build/bin."
            )
        logger.error(_state.error)
        return LlmServerState(**_state.__dict__)

    with _lock:
        _proc = proc
        _state.pid = proc.pid

    # Wait for server to be healthy
    deadline = time.time() + _STARTUP_TIMEOUT
    while time.time() < deadline:
        if proc.poll() is not None:
            with _lock:
                _state.status = "error"
                _state.error = f"llama-server exited early (code {proc.returncode})"
            logger.error(_state.error)
            return LlmServerState(**_state.__dict__)
        if _is_server_healthy():
            break
        time.sleep(_HEALTH_INTERVAL)
    else:
        proc.terminate()
        with _lock:
            _proc = None
            _state.status = "error"
            _state.error = "llama-server did not become healthy in time"
        logger.error(_state.error)
        return LlmServerState(**_state.__dict__)

    with _lock:
        _state.status = "running"
        _state.error = ""

    _update_env_url(f"http://127.0.0.1:{LLAMA_SERVER_PORT}/v1")
    logger.info(f"llama-server running on port {LLAMA_SERVER_PORT}, model={model_filename}")
    return LlmServerState(**_state.__dict__)


def unload_model() -> None:
    global _state, _proc

    with _lock:
        proc = _proc
        _proc = None
        _state = LlmServerState(status="stopped")

    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    _update_env_url("")
    logger.info("llama-server stopped")


# ── Local model management ────────────────────────────────────────────────────

def list_local_models() -> List[dict]:
    models = []
    for path in sorted(_models_dir().glob("*.gguf")):
        size_bytes = path.stat().st_size
        loaded = _state.model_filename == path.name and _state.status == "running"
        models.append({
            "filename": path.name,
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / (1024 * 1024), 1),
            "loaded": loaded,
        })
    return models


def delete_model(model_filename: str) -> None:
    if _state.model_filename == model_filename and _state.status == "running":
        raise RuntimeError("Cannot delete a model that is currently loaded. Unload it first.")
    path = _models_dir() / model_filename
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_filename}")
    path.unlink()
    logger.info(f"Deleted model: {model_filename}")


# ── HuggingFace search & download ─────────────────────────────────────────────

def search_huggingface(query: str, limit: int = 20) -> List[dict]:
    """Search HuggingFace Hub for GGUF models matching the query."""
    url = "https://huggingface.co/api/models"
    params = {
        "search": query,
        "filter": "gguf",
        "sort": "downloads",
        "direction": "-1",
        "limit": limit,
        "full": "false",
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        results = resp.json()
    except Exception as exc:
        logger.warning(f"HuggingFace search failed: {exc}")
        return []

    models = []
    for m in results:
        repo_id = m.get("modelId") or m.get("id", "")
        models.append({
            "repo_id": repo_id,
            "downloads": m.get("downloads", 0),
            "likes": m.get("likes", 0),
            "tags": m.get("tags", []),
            "pipeline_tag": m.get("pipeline_tag", ""),
        })
    return models


def get_model_files(repo_id: str) -> List[dict]:
    """List GGUF files available in a HuggingFace repository."""
    url = f"https://huggingface.co/api/models/{repo_id}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning(f"Failed to get model files for {repo_id}: {exc}")
        return []

    siblings = data.get("siblings", [])
    files = [
        {"filename": s["rfilename"], "size": s.get("size")}
        for s in siblings
        if s.get("rfilename", "").endswith(".gguf")
    ]
    return files


# Download progress tracking
_download_tasks: dict[str, dict] = {}  # key = "{repo_id}/{filename}"
_download_lock = threading.Lock()


def get_download_status(task_key: str) -> dict:
    with _download_lock:
        return _download_tasks.get(task_key, {"status": "not_found"})


def start_download(repo_id: str, filename: str) -> str:
    """Start async download. Returns task_key."""
    task_key = f"{repo_id}/{filename}"
    with _download_lock:
        if task_key in _download_tasks and _download_tasks[task_key]["status"] == "downloading":
            return task_key
        _download_tasks[task_key] = {"status": "downloading", "progress": 0, "error": ""}

    thread = threading.Thread(
        target=_download_worker,
        args=(repo_id, filename, task_key),
        daemon=True,
    )
    thread.start()
    return task_key


def _download_worker(repo_id: str, filename: str, task_key: str) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        with _download_lock:
            _download_tasks[task_key] = {
                "status": "error",
                "error": "huggingface_hub not installed. Run: pip install huggingface_hub",
            }
        return

    dest_dir = str(_models_dir())
    try:
        logger.info(f"Downloading {repo_id}/{filename} to {dest_dir}")
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=dest_dir,
            local_dir_use_symlinks=False,
        )
        # Move to top level of models dir if nested
        dest_path = _models_dir() / filename
        src = Path(local_path)
        if src != dest_path and src.exists():
            src.rename(dest_path)

        with _download_lock:
            _download_tasks[task_key] = {
                "status": "done",
                "filename": filename,
                "path": str(dest_path),
            }
        logger.info(f"Download complete: {filename}")
    except Exception as exc:
        logger.error(f"Download failed for {repo_id}/{filename}: {exc}")
        with _download_lock:
            _download_tasks[task_key] = {"status": "error", "error": str(exc)}
