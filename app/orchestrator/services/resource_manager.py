"""Shared runtime resource lifecycle manager.

This manager coordinates load/unload and lease-based usage tracking across
OCR/LLM/TTS resources so GPU/worker resources can be reclaimed safely.
"""
from __future__ import annotations

import gc
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from app.shared.logger import get_logger

logger = get_logger("resource_manager")


@dataclass
class ResourceState:
    engine: str
    model_id: str = ""
    loaded: bool = False
    device: str = ""
    memory: Dict[str, Any] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=dict)
    leases: Dict[str, int] = field(default_factory=dict)
    active_jobs: int = 0
    last_used_at: Optional[float] = None
    idle_timeout_seconds: int = 60
    idle_deadline_at: Optional[float] = None
    pending_unload: bool = False
    status: str = "stopped"
    last_error: str = ""


class ResourceManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._states: Dict[str, ResourceState] = {}
        self._loaders: Dict[str, Callable[[str, Dict[str, Any]], Dict[str, Any]]] = {}
        self._unloaders: Dict[str, Callable[[str, Dict[str, Any]], None]] = {}
        self._running = True
        self._watcher = threading.Thread(target=self._watch_idle, daemon=True, name="resource-idle-watcher")
        self._watcher.start()

    def register_engine(
        self,
        engine: str,
        *,
        loader: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None,
        unloader: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        idle_timeout_seconds: int = 60,
    ) -> None:
        with self._lock:
            if loader:
                self._loaders[engine] = loader
            if unloader:
                self._unloaders[engine] = unloader
            st = self._states.get(engine)
            if st is None:
                self._states[engine] = ResourceState(engine=engine, idle_timeout_seconds=idle_timeout_seconds)
            else:
                st.idle_timeout_seconds = idle_timeout_seconds

    def load(self, engine: str, model_id: str = "", options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        options = dict(options or {})
        with self._lock:
            st = self._states.setdefault(engine, ResourceState(engine=engine))
            st.status = "loading"
            st.last_error = ""
            if model_id:
                st.model_id = model_id
            if options:
                st.options.update(options)

        try:
            meta = {}
            loader = self._loaders.get(engine)
            if loader is not None:
                meta = dict(loader(model_id, options) or {})
            with self._lock:
                st = self._states[engine]
                st.loaded = True
                st.status = "running"
                st.last_used_at = time.time()
                st.pending_unload = False
                st.idle_deadline_at = None
                st.device = str(meta.get("device", st.device or options.get("device", "")))
                if "memory" in meta and isinstance(meta["memory"], dict):
                    st.memory = dict(meta["memory"])
            return self.get_status(engine)
        except Exception as exc:
            with self._lock:
                st = self._states[engine]
                st.status = "error"
                st.last_error = str(exc)
            raise

    def unload(self, engine: str, scope: str = "auto") -> bool:
        with self._lock:
            st = self._states.setdefault(engine, ResourceState(engine=engine))
            if st.active_jobs > 0 or sum(st.leases.values()) > 0:
                st.pending_unload = True
                logger.info(
                    "[%s] unload deferred (scope=%s): active_jobs=%s leases=%s",
                    engine,
                    scope,
                    st.active_jobs,
                    sum(st.leases.values()),
                )
                return False
            already_loaded = st.loaded
            st.status = "stopping"
            st.pending_unload = False
            st.idle_deadline_at = None

        try:
            unloader = self._unloaders.get(engine)
            if unloader:
                unloader(scope, dict(st.options))
        finally:
            self._post_unload_cleanup(engine)
            with self._lock:
                st = self._states[engine]
                st.loaded = False
                st.status = "stopped"
                st.device = ""
                st.memory = {}
                st.model_id = ""
                st.options = {}
                st.last_used_at = time.time()
        return already_loaded

    def acquire_lease(self, engine: str, reason: str, model_id: str = "", options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        options = dict(options or {})
        with self._lock:
            st = self._states.setdefault(engine, ResourceState(engine=engine))
            if options.get("idle_timeout_seconds") is not None:
                try:
                    st.idle_timeout_seconds = int(options["idle_timeout_seconds"])
                except Exception:
                    pass
            if not st.loaded:
                pass
        if not self._states[engine].loaded:
            self.load(engine, model_id, options)

        with self._lock:
            st = self._states[engine]
            st.leases[reason] = st.leases.get(reason, 0) + 1
            st.active_jobs += 1
            st.last_used_at = time.time()
            st.pending_unload = False
            st.idle_deadline_at = None
            return self.get_status(engine)

    def release_lease(self, engine: str, reason: str) -> Dict[str, Any]:
        with self._lock:
            st = self._states.setdefault(engine, ResourceState(engine=engine))
            if reason in st.leases:
                st.leases[reason] = max(0, st.leases[reason] - 1)
                if st.leases[reason] == 0:
                    del st.leases[reason]
            st.active_jobs = max(0, st.active_jobs - 1)
            st.last_used_at = time.time()
            total = sum(st.leases.values())
            if total == 0:
                st.idle_deadline_at = time.time() + max(0, st.idle_timeout_seconds)
            return self.get_status(engine)

    def get_status(self, engine: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            if engine is not None:
                st = self._states.setdefault(engine, ResourceState(engine=engine))
                return self._state_to_dict(st)
            return {k: self._state_to_dict(v) for k, v in sorted(self._states.items())}

    def _state_to_dict(self, st: ResourceState) -> Dict[str, Any]:
        now = time.time()
        idle_seconds = (now - st.last_used_at) if st.last_used_at else None
        ttl = (st.idle_deadline_at - now) if st.idle_deadline_at else None
        return {
            "engine": st.engine,
            "status": st.status,
            "loaded": st.loaded,
            "model_id": st.model_id,
            "device": st.device,
            "memory": st.memory,
            "options": dict(st.options),
            "leases": dict(st.leases),
            "lease_total": sum(st.leases.values()),
            "active_jobs": st.active_jobs,
            "last_used_at": st.last_used_at,
            "idle_seconds": round(idle_seconds, 2) if idle_seconds is not None else None,
            "idle_timeout_seconds": st.idle_timeout_seconds,
            "idle_deadline_at": st.idle_deadline_at,
            "idle_remaining_seconds": round(ttl, 2) if ttl is not None else None,
            "pending_unload": st.pending_unload,
            "last_error": st.last_error,
        }

    def _watch_idle(self) -> None:
        while self._running:
            time.sleep(1.0)
            due: list[str] = []
            with self._lock:
                now = time.time()
                for eng, st in self._states.items():
                    if not st.loaded:
                        continue
                    if st.active_jobs > 0 or sum(st.leases.values()) > 0:
                        continue
                    if st.idle_deadline_at is not None and now >= st.idle_deadline_at:
                        due.append(eng)
            for eng in due:
                try:
                    logger.info("[%s] idle timeout reached, unloading", eng)
                    self.unload(eng, scope="idle_timeout")
                except Exception as exc:
                    logger.warning("[%s] idle unload failed: %s", eng, exc)

    @staticmethod
    def _post_unload_cleanup(engine: str) -> None:
        gc.collect()
        lower = engine.lower()
        if "torch" in lower or "llm" in lower or "qwen" in lower or "paddle" in lower:
            try:
                import torch  # type: ignore

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, "ipc_collect"):
                        torch.cuda.ipc_collect()
            except Exception:
                pass
        if "paddle" in lower:
            try:
                import paddle  # type: ignore

                if hasattr(paddle.device.cuda, "empty_cache"):
                    paddle.device.cuda.empty_cache()
            except Exception:
                pass


def _supervisorctl(action: str, program: str) -> None:
    sock = os.environ.get("SUPERVISOR_SOCKET", "/var/run/supervisor.sock")
    cmd = ["supervisorctl", "-s", f"unix://{sock}", action, program]
    subprocess.run(cmd, check=False, capture_output=True, text=True)


def _tts_loader(model_id: str, options: Dict[str, Any]) -> Dict[str, Any]:
    worker = options.get("worker_type") or model_id
    if options.get("ensure_supervisor", True):
        _supervisorctl("start", f"tts_{worker}")
    return {"device": options.get("device", "")}


def _tts_unloader(scope: str, options: Dict[str, Any]) -> None:
    worker = options.get("worker_type")
    if worker and options.get("stop_on_unload", True):
        _supervisorctl("stop", f"tts_{worker}")


def _paddle_loader(model_id: str, options: Dict[str, Any]) -> Dict[str, Any]:
    from app.orchestrator.services.ocr.paddleocr_engine import PaddleOCREngine

    device = str(options.get("device") or os.environ.get("OCR_PADDLE_DEVICE", "gpu:0"))
    use_layout = bool(options.get("use_layout", False))
    PaddleOCREngine.configure(device=device, use_layout=use_layout)
    # Warm up lazy singleton if needed
    PaddleOCREngine().is_available()
    return {"device": device}


def _paddle_unloader(scope: str, options: Dict[str, Any]) -> None:
    from app.orchestrator.services.ocr.paddleocr_engine import PaddleOCREngine

    PaddleOCREngine.release_resources()


def _llm_loader(model_id: str, options: Dict[str, Any]) -> Dict[str, Any]:
    from app.orchestrator.services import llm_manager

    if llm_manager.get_server_status().status == "running":
        return {"device": "cuda" if options.get("n_gpu_layers", -1) != 0 else "cpu"}

    model = model_id or llm_manager.get_main_model()
    if not model:
        models = llm_manager.list_local_models()
        if models:
            model = models[0]["filename"]
    if not model:
        raise RuntimeError("No LLM model available for loading")

    settings = llm_manager.get_model_setting(model)
    settings.update(options)
    llm_manager.load_model(
        model_filename=model,
        n_gpu_layers=int(settings.get("n_gpu_layers", -1)),
        ctx_size=int(settings.get("ctx_size", 4096)),
        batch_size=int(settings.get("batch_size", 512)),
        threads=int(settings.get("threads", -1)),
        flash_attn=bool(settings.get("flash_attn", False)),
    )
    return {"device": "cuda" if int(settings.get("n_gpu_layers", -1)) != 0 else "cpu"}


def _llm_unloader(scope: str, options: Dict[str, Any]) -> None:
    from app.orchestrator.services import llm_manager

    llm_manager.unload_model()


_manager = ResourceManager()

# Defaults: tts=30s, ocr=60s, llm=300s (overridable by env)
_manager.register_engine(
    "llm",
    loader=_llm_loader,
    unloader=_llm_unloader,
    idle_timeout_seconds=int(os.environ.get("LLM_IDLE_TIMEOUT_SECONDS", "300")),
)
_manager.register_engine(
    "paddleocr",
    loader=_paddle_loader,
    unloader=_paddle_unloader,
    idle_timeout_seconds=int(os.environ.get("OCR_IDLE_TIMEOUT_SECONDS", "60")),
)
for _w in ("base", "custom", "design"):
    _manager.register_engine(
        f"tts_{_w}",
        loader=_tts_loader,
        unloader=_tts_unloader,
        idle_timeout_seconds=int(os.environ.get("TTS_IDLE_TIMEOUT_SECONDS", "30")),
    )


def load(engine: str, model_id: str = "", options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return _manager.load(engine, model_id, options)


def unload(engine: str, scope: str = "auto") -> bool:
    return _manager.unload(engine, scope)


def acquire_lease(engine: str, reason: str, model_id: str = "", options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return _manager.acquire_lease(engine, reason, model_id=model_id, options=options)


def release_lease(engine: str, reason: str) -> Dict[str, Any]:
    return _manager.release_lease(engine, reason)


def get_status(engine: Optional[str] = None) -> Dict[str, Any]:
    return _manager.get_status(engine)
