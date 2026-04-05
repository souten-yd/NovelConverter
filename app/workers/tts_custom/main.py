"""TTS Worker Custom – custom voice / fixed speaker mode."""
from __future__ import annotations

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.workers.tts_custom.synthesizer import MockSynthesizer, Qwen3CustomSynthesizer
from app.shared.schemas import SynthesizeRequest, SynthesizeResponse, WorkerHealth
from app.shared.logger import get_logger

logger = get_logger("worker.custom")

WORKER_ID = "tts_worker_custom"
WORKER_TYPE = "custom"
PORT = int(os.environ.get("TTS_CUSTOM_PORT", 8002))
USE_REAL_MODEL = os.environ.get("TTS_CUSTOM_USE_REAL", "false").lower() == "true"

app = FastAPI(title="TTS Worker Custom (Custom Voice)", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_synth = None


def get_synth():
    global _synth
    if _synth is None:
        if USE_REAL_MODEL:
            _synth = Qwen3CustomSynthesizer()
        else:
            _synth = MockSynthesizer(worker_id=WORKER_ID)
            logger.info("Using MOCK synthesizer (set TTS_CUSTOM_USE_REAL=true for real model)")
    return _synth


@app.get("/health", response_model=WorkerHealth)
def health():
    return WorkerHealth(
        status="ok",
        worker_id=WORKER_ID,
        worker_type=WORKER_TYPE,
        model_loaded=get_synth().is_loaded(),
    )


@app.get("/models")
def list_models():
    return {"worker_id": WORKER_ID, "worker_type": WORKER_TYPE, "models": get_synth().available_models()}


@app.get("/speakers")
def list_speakers():
    synth = get_synth()
    speakers = synth.available_speakers() if hasattr(synth, "available_speakers") else []
    return {"speakers": speakers}


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
        "speakers": synth.available_speakers() if hasattr(synth, "available_speakers") else [],
    }
    if hasattr(synth, "get_load_error"):
        result["load_error"] = synth.get_load_error()
    if hasattr(synth, "get_runtime_status"):
        result.update(synth.get_runtime_status())
    try:
        from app.shared.model_manager import get_models_status
        tts_status = get_models_status()
        result["tokenizer_status"] = tts_status.get("tokenizer", {}).get("status", "unknown")
        result["model_status"] = tts_status.get("custom", {}).get("status", "unknown")
    except Exception:
        pass
    return result


@app.post("/synthesize", response_model=SynthesizeResponse)
def synthesize(req: SynthesizeRequest):
    logger.info(f"[custom] synthesize: {req.text[:40]!r} speaker={req.speaker}")
    return get_synth().synthesize(req)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.workers.tts_custom.main:app", host="0.0.0.0", port=PORT, reload=False)
