"""TTS Worker Design – voice design mode."""
from __future__ import annotations

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.workers.tts_design.synthesizer import MockSynthesizer, Qwen3DesignSynthesizer
from app.shared.schemas import SynthesizeRequest, SynthesizeResponse, WorkerHealth
from app.shared.logger import get_logger

logger = get_logger("worker.design")

WORKER_ID = "tts_worker_design"
WORKER_TYPE = "design"
PORT = int(os.environ.get("TTS_DESIGN_PORT", 8003))
USE_REAL_MODEL = os.environ.get("TTS_DESIGN_USE_REAL", "false").lower() == "true"
PRELOAD_ON_STARTUP = os.environ.get("TTS_DESIGN_PRELOAD_ON_STARTUP", "false").lower() == "true"

app = FastAPI(title="TTS Worker Design (Voice Design)", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_synth = None


def get_synth():
    global _synth
    if _synth is None:
        if USE_REAL_MODEL:
            _synth = Qwen3DesignSynthesizer()
        else:
            _synth = MockSynthesizer(worker_id=WORKER_ID)
            logger.info("Using MOCK synthesizer (set TTS_DESIGN_USE_REAL=true for real model)")
    return _synth


@app.on_event("startup")
def startup_preflight():
    if not USE_REAL_MODEL:
        return
    if not PRELOAD_ON_STARTUP:
        logger.info("TTS Design preload disabled (set TTS_DESIGN_PRELOAD_ON_STARTUP=true to enable eager load).")
        return
    synth = get_synth()
    if not synth.is_loaded():
        err = synth.get_load_error() if hasattr(synth, "get_load_error") else "unknown error"
        logger.error("TTS Design preflight failed: %s", err)
        raise RuntimeError(f"TTS Design preflight failed: {err}")


@app.get("/health", response_model=WorkerHealth)
def health():
    loaded = _synth.is_loaded() if _synth is not None else False
    return WorkerHealth(
        status="ok",
        worker_id=WORKER_ID,
        worker_type=WORKER_TYPE,
        model_loaded=loaded,
    )


@app.get("/models")
def list_models():
    return {"worker_id": WORKER_ID, "worker_type": WORKER_TYPE, "models": get_synth().available_models()}


@app.post("/warmup")
def warmup():
    synth = get_synth()
    synth.warmup()
    loaded = synth.is_loaded()
    error = synth.get_load_error() if hasattr(synth, "get_load_error") else None
    return {"status": "ok" if loaded else "error", "model_loaded": loaded, "error": error}


@app.get("/model_status")
def model_status():
    synth = get_synth()
    result = {
        "worker_id": WORKER_ID,
        "worker_type": WORKER_TYPE,
        "use_real_model": USE_REAL_MODEL,
        "model_loaded": synth.is_loaded(),
    }
    if hasattr(synth, "get_load_error"):
        result["load_error"] = synth.get_load_error()
    if hasattr(synth, "get_runtime_status"):
        result.update(synth.get_runtime_status())
    try:
        from app.shared.model_manager import get_models_status
        tts_status = get_models_status()
        result["tokenizer_status"] = tts_status.get("tokenizer", {}).get("status", "unknown")
        result["model_status"] = tts_status.get("design", {}).get("status", "unknown")
    except Exception:
        pass
    return result


@app.get("/diagnostics")
def diagnostics():
    synth = get_synth()
    status = synth.get_runtime_status() if hasattr(synth, "get_runtime_status") else {}
    status.update(
        {
            "worker_id": WORKER_ID,
            "worker_type": WORKER_TYPE,
            "use_real_model": USE_REAL_MODEL,
            "model_loaded": synth.is_loaded(),
            "load_error": synth.get_load_error() if hasattr(synth, "get_load_error") else None,
        }
    )
    return status


@app.post("/synthesize", response_model=SynthesizeResponse)
def synthesize(req: SynthesizeRequest):
    logger.info(f"[design] synthesize: {req.text[:40]!r} desc={req.voice_description!r}")
    return get_synth().synthesize(req)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.workers.tts_design.main:app", host="0.0.0.0", port=PORT, reload=False)
