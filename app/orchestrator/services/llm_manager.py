"""LLM server manager – launch/stop llama.cpp server, manage GGUF models."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import requests

from app.shared.logger import get_logger
from app.shared.paths import get_models_root

logger = get_logger("llm_manager")

LLAMA_SERVER_PORT = int(os.environ.get("LLAMA_SERVER_PORT", "8080"))
MODELS_DIR_NAME = "llm"

_STARTUP_TIMEOUT = 60  # seconds to wait for llama-server to become ready
_HEALTH_INTERVAL = 2   # seconds between health-check polls
_AUTO_UNLOAD_CHECK_INTERVAL = 30  # seconds between auto-unload checks

# Default per-model settings shape
_DEFAULT_MODEL_SETTINGS: dict = {
    "n_gpu_layers": -1,
    "ctx_size": 4096,
    "batch_size": 512,
    "threads": -1,      # -1 = llama.cpp default
    "flash_attn": False,
    "auto_unload_seconds": 0,  # 0 = disabled
    "is_main_model": False,
}


# ── State ─────────────────────────────────────────────────────────────────────

@dataclass
class LlmServerState:
    status: str = "stopped"   # stopped / loading / running / error
    model_filename: str = ""
    model_path: str = ""
    port: int = LLAMA_SERVER_PORT
    pid: Optional[int] = None
    error: str = ""
    stderr: str = ""
    last_used_at: Optional[float] = None   # unix timestamp
    load_params: dict = field(default_factory=dict)


_state = LlmServerState()
_proc: Optional[subprocess.Popen] = None
_lock = threading.Lock()
_auto_unload_thread: Optional[threading.Thread] = None
_auto_unload_running = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _models_dir() -> Path:
    d = get_models_root() / MODELS_DIR_NAME
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


# ── Model settings persistence ────────────────────────────────────────────────

def _settings_path() -> Path:
    return _models_dir() / "llm_settings.json"


def load_model_settings() -> Dict[str, dict]:
    """Load per-model settings from disk. Returns dict: filename → settings."""
    p = _settings_path()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning(f"Failed to load llm_settings.json: {e}")
        return {}


def save_model_settings(settings: Dict[str, dict]) -> None:
    """Persist per-model settings to disk."""
    try:
        _settings_path().write_text(
            json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as e:
        logger.warning(f"Failed to save llm_settings.json: {e}")


def get_model_setting(model_filename: str) -> dict:
    """Get settings for a single model (merged with defaults)."""
    all_settings = load_model_settings()
    stored = all_settings.get(model_filename, {})
    merged = dict(_DEFAULT_MODEL_SETTINGS)
    merged.update(stored)
    return merged


def update_model_setting(model_filename: str, updates: dict) -> dict:
    """Update settings for a model and persist. Returns new merged settings."""
    all_settings = load_model_settings()
    existing = all_settings.get(model_filename, {})
    existing.update(updates)
    # If this model is set as main, unset others
    if updates.get("is_main_model"):
        for fn in list(all_settings.keys()):
            if fn != model_filename:
                all_settings[fn]["is_main_model"] = False
    all_settings[model_filename] = existing
    save_model_settings(all_settings)
    return get_model_setting(model_filename)


def get_main_model() -> Optional[str]:
    """Return filename of the model marked as main (or None)."""
    for fn, settings in load_model_settings().items():
        if settings.get("is_main_model"):
            return fn
    return None


# ── Auto-unload watcher ───────────────────────────────────────────────────────

def _auto_unload_watcher() -> None:
    global _auto_unload_running
    logger.info("Auto-unload watcher started")
    while _auto_unload_running:
        time.sleep(_AUTO_UNLOAD_CHECK_INTERVAL)
        with _lock:
            if _state.status != "running":
                continue
            model_fn = _state.model_filename
            last_used = _state.last_used_at

        setting = get_model_setting(model_fn)
        idle_timeout = int(setting.get("auto_unload_seconds", 0))
        if idle_timeout <= 0:
            continue  # disabled for this model
        if last_used is None:
            continue
        idle_secs = time.time() - last_used
        if idle_secs >= idle_timeout:
            logger.info(
                f"Auto-unloading model {model_fn!r} after {idle_secs:.0f}s idle "
                f"(timeout={idle_timeout}s)"
            )
            unload_model()


def start_auto_unload_watcher() -> None:
    global _auto_unload_thread, _auto_unload_running
    if _auto_unload_thread and _auto_unload_thread.is_alive():
        return  # already running
    _auto_unload_running = True
    _auto_unload_thread = threading.Thread(
        target=_auto_unload_watcher, daemon=True, name="llm-auto-unload"
    )
    _auto_unload_thread.start()


def touch_last_used() -> None:
    """Record that the LLM was just used. Call from speaker_segmenter etc."""
    with _lock:
        if _state.status == "running":
            _state.last_used_at = time.time()


# ── Public API ────────────────────────────────────────────────────────────────

def get_server_status() -> LlmServerState:
    global _state, _proc
    with _lock:
        if _state.status == "running" and _proc is not None:
            if _proc.poll() is not None:
                # process died
                _state.status = "error"
                _state.error = f"llama-server exited with code {_proc.returncode}"
                _state.stderr = ""
                _proc = None
        return LlmServerState(**_state.__dict__)


def get_server_status_dict() -> dict:
    """Full status dict suitable for API response, includes computed fields."""
    state = get_server_status()
    now = time.time()
    idle_seconds = round(now - state.last_used_at, 1) if state.last_used_at else None
    setting = get_model_setting(state.model_filename) if state.model_filename else {}
    return {
        "status": state.status,
        "model_filename": state.model_filename,
        "model_path": state.model_path,
        "port": state.port,
        "pid": state.pid,
        "error": state.error,
        "stderr": state.stderr,
        "binary_available": check_binary_available(),
        "last_used_at": state.last_used_at,
        "idle_seconds": idle_seconds,
        "load_params": state.load_params,
        "auto_unload_seconds": setting.get("auto_unload_seconds", 0),
    }


def load_model(
    model_filename: str,
    n_gpu_layers: int = -1,
    ctx_size: int = 4096,
    batch_size: int = 512,
    threads: int = -1,
    flash_attn: bool = False,
) -> LlmServerState:
    global _state, _proc

    with _lock:
        if _state.status in ("loading", "running"):
            raise RuntimeError("A model is already loaded. Unload it first.")

        model_path = _models_dir() / model_filename
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_filename}")

        _load_params = {
            "n_gpu_layers": n_gpu_layers,
            "ctx_size": ctx_size,
            "batch_size": batch_size,
            "threads": threads,
            "flash_attn": flash_attn,
        }
        _state = LlmServerState(
            status="loading",
            model_filename=model_filename,
            model_path=str(model_path),
            port=LLAMA_SERVER_PORT,
            load_params=_load_params,
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
        "--batch-size", str(batch_size),
        "--host", "127.0.0.1",
    ]
    if threads > 0:
        cmd += ["--threads", str(threads)]
    if flash_attn:
        cmd += ["--flash-attn"]
    logger.info(f"Launching llama-server: {' '.join(cmd)}")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
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
            stdout_text, stderr_text = proc.communicate()
            stderr_text = (stderr_text or "").strip()
            stdout_text = (stdout_text or "").strip()
            combined_error_log = stderr_text or stdout_text
            with _lock:
                _state.status = "error"
                _state.stderr = combined_error_log
                if proc.returncode == -6 and stderr_text:
                    _state.error = (
                        f"llama-server exited early (code -6). stderr:\n{stderr_text}"
                    )
                else:
                    _state.error = f"llama-server exited early (code {proc.returncode})"
            logger.error(_state.error)
            if combined_error_log:
                logger.error("llama-server startup output:\n%s", combined_error_log)
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
        _state.last_used_at = time.time()

    _update_env_url(f"http://127.0.0.1:{LLAMA_SERVER_PORT}/v1")
    logger.info(f"llama-server running on port {LLAMA_SERVER_PORT}, model={model_filename}")

    # Start auto-unload watcher (no-op if already running)
    start_auto_unload_watcher()

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
    all_settings = load_model_settings()
    models = []
    for path in sorted(_models_dir().glob("*.gguf")):
        size_bytes = path.stat().st_size
        loaded = _state.model_filename == path.name and _state.status == "running"
        settings = dict(_DEFAULT_MODEL_SETTINGS)
        settings.update(all_settings.get(path.name, {}))
        models.append({
            "filename": path.name,
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / (1024 * 1024), 1),
            "loaded": loaded,
            "settings": settings,
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


# ── Reference-counted acquire / release ──────────────────────────────────────

_refcount: dict[str, int] = {}
_refcount_lock = threading.Lock()
_idle_unload_delay = int(os.environ.get("LLM_IDLE_TIMEOUT_SECONDS", "300"))


def acquire(reason: str) -> None:
    """Request LLM lease via shared resource manager and increment local refcount."""
    logger.info(f"[LLM] load requested: {reason}")
    from app.orchestrator.services.resource_manager import acquire_lease

    with _refcount_lock:
        _refcount[reason] = _refcount.get(reason, 0) + 1
    acquire_lease(
        "llm",
        reason=reason,
        options={"idle_timeout_seconds": _idle_unload_delay},
    )
    touch_last_used()


def release(reason: str) -> None:
    """Release LLM lease via shared resource manager and decrement local refcount."""
    logger.info(f"[LLM] release: {reason}")
    from app.orchestrator.services.resource_manager import release_lease

    with _refcount_lock:
        if reason in _refcount:
            _refcount[reason] = max(0, _refcount[reason] - 1)
            if _refcount[reason] == 0:
                del _refcount[reason]
    release_lease("llm", reason=reason)


def _wait_for_running(timeout: float = 60) -> bool:
    """Block until LLM server is running or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if get_server_status().status == "running":
            return True
        time.sleep(2)
    return False


def force_unload() -> None:
    """Force-unload LLM regardless of refcount."""
    logger.info("[LLM] force unload requested")
    from app.orchestrator.services.resource_manager import unload

    with _refcount_lock:
        _refcount.clear()
    unload("llm", scope="force")


def is_llm_loaded() -> bool:
    return get_server_status().status == "running"


def get_refcount_status() -> dict:
    """Return current refcount state for UI/API display."""
    from app.orchestrator.services.resource_manager import get_status

    with _refcount_lock:
        counts = dict(_refcount)
        total = sum(counts.values())
    status = get_server_status()
    lifecycle = get_status("llm")
    return {
        "refcounts": counts,
        "total_refcount": total,
        "llm_status": status.status,
        "model": status.model_filename,
        "idle_unload_delay": _idle_unload_delay,
        "lifecycle": lifecycle,
    }


# ── Download worker (existing) ──────────────────────────────────────────────

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
