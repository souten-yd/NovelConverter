"""Runtime control/status APIs for OCR/TTS/LLM/VLM workers."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.orchestrator.services import llm_manager
from app.orchestrator.services.resource_manager import get_status as get_resource_status, unload as unload_resource
from app.shared.database import get_db
from app.shared.models import Project, Speaker, VoiceProfile
from app.shared.runtime_backends import load_runtime_status

router = APIRouter(prefix="/api/runtime", tags=["runtime_control"])


# --- Response normalization helpers -----------------------------------------

def _ok(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"ok": True, **payload}


def _error_payload(*, reason: str, code: str, detail: Any) -> Dict[str, Any]:
    return {
        "ok": False,
        "reason": reason,
        "code": code,
        "detail": detail,
    }


def _raise(status_code: int, *, reason: str, code: str, detail: Any) -> None:
    raise HTTPException(status_code=status_code, detail=_error_payload(reason=reason, code=code, detail=detail))


def _fetch_json(url: str, *, timeout: int = 5) -> Dict[str, Any]:
    r = requests.get(url, timeout=timeout)
    try:
        body = r.json()
    except Exception:
        body = {"raw": r.text}
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {body}")
    return body


def _engine_view(st: Optional[Dict[str, Any]], *, fallback_engine: str) -> Dict[str, Any]:
    st = st or {}
    return {
        "engine": st.get("engine", fallback_engine),
        "status": st.get("status", "unknown"),
        "loaded": bool(st.get("loaded", False)),
        "current_model": st.get("model_id") or None,
        "device": st.get("device") or None,
        "memory": st.get("memory") or {},
        "idle_timer": {
            "seconds": st.get("idle_seconds"),
            "timeout_seconds": st.get("idle_timeout_seconds"),
            "remaining_seconds": st.get("idle_remaining_seconds"),
            "deadline_at": st.get("idle_deadline_at"),
        },
        "active_jobs": st.get("active_jobs", 0),
        "lease_total": st.get("lease_total", 0),
        "pending_unload": bool(st.get("pending_unload", False)),
        "last_error": st.get("last_error") or None,
    }


def _worker_status_urls() -> Dict[str, str]:
    return {
        "base": os.environ.get("TTS_BASE_URL", "http://localhost:8001"),
        "custom": os.environ.get("TTS_CUSTOM_URL", "http://localhost:8002"),
        "design": os.environ.get("TTS_DESIGN_URL", "http://localhost:8003"),
    }


@router.get("/status")
def unified_runtime_status(
    db: Session = Depends(get_db),
    project_id: Optional[str] = Query(default=None),
    lite: bool = Query(
        default=False,
        description="Skip heavyweight runtime probes (recommended for project page initial load).",
    ),
) -> Dict[str, Any]:
    """Unified runtime status for OCR/TTS/LLM/VLM in a normalized format."""
    lifecycle = get_resource_status()

    # LLM / VLM status
    llm_server = llm_manager.get_server_status_dict()
    llm_lifecycle = lifecycle.get("llm", {})
    llm_status = {
        **_engine_view(llm_lifecycle, fallback_engine="llm"),
        "status": llm_server.get("status", llm_lifecycle.get("status", "unknown")),
        "loaded": llm_server.get("status") in ("running", "loading") or bool(llm_lifecycle.get("loaded", False)),
        "current_model": llm_server.get("model_filename") or llm_lifecycle.get("model_id") or None,
        "port": llm_server.get("port"),
        "pid": llm_server.get("pid"),
        "binary_available": llm_server.get("binary_available"),
        "last_error": llm_server.get("error") or llm_lifecycle.get("last_error") or None,
    }

    # OCR backends (paddle + ndlocr + qwen-vl)
    ocr_status: Dict[str, Any] = {}
    paddle_lifecycle = lifecycle.get("paddleocr", {})
    if lite:
        ocr_status["paddleocr"] = _engine_view(paddle_lifecycle, fallback_engine="paddleocr")
        ocr_status["paddleocr"].update({"available": None, "resolved_device": None, "paddleocr_version": None})
        ocr_status["ndlocr_lite"] = _engine_view(lifecycle.get("ndlocr_lite", {}), fallback_engine="ndlocr_lite")
        ocr_status["qwen_vl"] = _engine_view(lifecycle.get("qwen_vl", {}), fallback_engine="qwen_vl")
    else:
        try:
            from app.orchestrator.services.ocr.paddleocr_engine import PaddleOCREngine

            paddle_rt = PaddleOCREngine.get_runtime_status()
            ocr_status["paddleocr"] = {
                **_engine_view(paddle_lifecycle, fallback_engine="paddleocr"),
                "available": paddle_rt.get("available"),
                "resolved_device": paddle_rt.get("resolved_device"),
                "paddleocr_version": paddle_rt.get("paddleocr_version"),
                "last_error": paddle_rt.get("last_error") or paddle_lifecycle.get("last_error") or None,
            }
        except Exception as exc:
            ocr_status["paddleocr"] = {
                **_engine_view(paddle_lifecycle, fallback_engine="paddleocr"),
                "available": False,
                "last_error": str(exc),
            }

        try:
            from app.orchestrator.services.ocr.ndlocr_lite_engine import get_ndlocr_status

            ndl = get_ndlocr_status()
            ocr_status["ndlocr_lite"] = {
                "engine": "ndlocr_lite",
                "status": "running" if ndl.get("runtime_ready") else "error",
                "loaded": bool(ndl.get("runtime_ready")),
                "current_model": ndl.get("model_path") if ndl.get("model_files_present") else None,
                "device": "cpu",
                "memory": {},
                "idle_timer": {"seconds": None, "timeout_seconds": None, "remaining_seconds": None, "deadline_at": None},
                "active_jobs": 0,
                "lease_total": 0,
                "pending_unload": False,
                "installed": ndl.get("installed"),
                "cli_functional": ndl.get("cli_functional"),
                "last_error": ndl.get("error_message") or None,
            }
        except Exception as exc:
            ocr_status["ndlocr_lite"] = {
                "engine": "ndlocr_lite",
                "status": "error",
                "loaded": False,
                "current_model": None,
                "device": "cpu",
                "memory": {},
                "idle_timer": {"seconds": None, "timeout_seconds": None, "remaining_seconds": None, "deadline_at": None},
                "active_jobs": 0,
                "lease_total": 0,
                "pending_unload": False,
                "last_error": str(exc),
            }

        try:
            from app.orchestrator.services.ocr.qwen_vl_engine import get_qwen_vl_runtime_status

            ocr_status["qwen_vl"] = get_qwen_vl_runtime_status()
        except Exception as exc:
            ocr_status["qwen_vl"] = {
                "engine": "qwen_vl",
                "status": "error",
                "loaded": False,
                "current_model": None,
                "device": None,
                "memory": {},
                "idle_timer": {"seconds": None, "timeout_seconds": None, "remaining_seconds": None, "deadline_at": None},
                "active_jobs": 0,
                "lease_total": 0,
                "pending_unload": False,
                "mode": "unknown",
                "last_error": str(exc),
            }

    # TTS workers + lifecycle states
    tts_status: Dict[str, Any] = {}
    for worker_type, base_url in _worker_status_urls().items():
        lifecycle_key = f"tts_{worker_type}"
        st = _engine_view(lifecycle.get(lifecycle_key, {}), fallback_engine=lifecycle_key)
        if lite:
            st.update({"alive": None, "url": base_url, "health": {}, "model_status": {}})
        else:
            try:
                health = _fetch_json(f"{base_url}/health")
                model_status = _fetch_json(f"{base_url}/model_status")
                st.update(
                    {
                        "alive": True,
                        "url": base_url,
                        "worker_id": health.get("worker_id"),
                        "health": health,
                        "model_status": model_status,
                        "current_model": model_status.get("model_path") or st.get("current_model"),
                        "device": model_status.get("device") or st.get("device"),
                        "last_error": model_status.get("last_error") or st.get("last_error"),
                    }
                )
            except Exception as exc:
                st.update(
                    {
                        "alive": False,
                        "url": base_url,
                        "health": {},
                        "model_status": {},
                        "last_error": str(exc),
                    }
                )
        tts_status[worker_type] = st

    # Optional project scoped voice preset usage
    project_summary: Optional[Dict[str, Any]] = None
    if project_id:
        project = db.get(Project, project_id)
        if not project:
            _raise(
                404,
                reason="Project not found",
                code="project_not_found",
                detail={"project_id": project_id},
            )

        speaker_count = db.query(Speaker).filter(Speaker.project_id == project_id).count()
        profiles = (
            db.query(VoiceProfile)
            .join(Speaker, Speaker.id == VoiceProfile.speaker_id)
            .filter(Speaker.project_id == project_id)
            .all()
        )
        preset_usage: Dict[str, int] = {}
        worker_usage: Dict[str, int] = {"base": 0, "custom": 0, "design": 0}
        for vp in profiles:
            worker = (vp.worker_type or "custom").strip().lower()
            if worker not in worker_usage:
                worker_usage[worker] = 0
            worker_usage[worker] += 1
            preset_key = vp.preset_id or "(manual)"
            preset_usage[preset_key] = preset_usage.get(preset_key, 0) + 1

        project_summary = {
            "project_id": project_id,
            "speaker_count": speaker_count,
            "profile_count": len(profiles),
            "preset_usage": [
                {"preset_id": k, "count": v}
                for k, v in sorted(preset_usage.items(), key=lambda x: (-x[1], x[0]))
            ],
            "worker_usage": worker_usage,
        }

    return _ok(
        {
            "runtime": load_runtime_status(),
            "engines": {
                "ocr": ocr_status,
                "tts": tts_status,
                "llm": llm_status,
                "vlm": ocr_status.get("qwen_vl", {}),
            },
            "project": project_summary,
        }
    )


@router.post("/stop/ocr")
def stop_ocr_worker() -> Dict[str, Any]:
    """Manual stop for OCR workers/backends."""
    details: Dict[str, Any] = {}
    # paddle lifecycle + singleton release
    try:
        was_loaded = unload_resource("paddleocr", scope="manual_stop")
        details["paddleocr"] = {"requested": True, "previously_loaded": was_loaded}
    except Exception as exc:
        details["paddleocr"] = {"requested": True, "error": str(exc)}

    # qwen-vl local model release
    try:
        from app.orchestrator.services.ocr.qwen_vl_engine import release_qwen_vl_resources

        details["qwen_vl"] = release_qwen_vl_resources()
    except Exception as exc:
        details["qwen_vl"] = {"released": False, "error": str(exc)}

    return _ok({"target": "ocr", "result": details})


@router.post("/stop/tts")
def stop_tts_worker(worker_type: Optional[str] = Query(default=None, description="base/custom/design (omit for all)")) -> Dict[str, Any]:
    """Manual stop for TTS workers."""
    targets = [worker_type.strip().lower()] if worker_type else ["base", "custom", "design"]
    allowed = {"base", "custom", "design"}
    bad = [w for w in targets if w not in allowed]
    if bad:
        _raise(
            400,
            reason="Invalid worker_type",
            code="invalid_worker_type",
            detail={"worker_type": bad[0], "allowed": sorted(allowed)},
        )

    stopped: Dict[str, Any] = {}
    for w in targets:
        engine = f"tts_{w}"
        try:
            was_loaded = unload_resource(engine, scope="manual_stop")
            stopped[w] = {"requested": True, "previously_loaded": was_loaded}
        except Exception as exc:
            stopped[w] = {"requested": True, "error": str(exc)}

    return _ok({"target": "tts", "result": stopped})


@router.post("/stop/llm-vlm")
def stop_llm_vlm_worker() -> Dict[str, Any]:
    """Manual stop for LLM server and VLM local resources."""
    result: Dict[str, Any] = {}

    try:
        llm_manager.force_unload()
        result["llm"] = {"requested": True, "stopped": True}
    except Exception as exc:
        result["llm"] = {"requested": True, "stopped": False, "error": str(exc)}

    try:
        from app.orchestrator.services.ocr.qwen_vl_engine import release_qwen_vl_resources

        result["vlm"] = release_qwen_vl_resources()
    except Exception as exc:
        result["vlm"] = {"released": False, "error": str(exc)}

    return _ok({"target": "llm_vlm", "result": result})
