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


@app.post("/warmup")
def warmup():
    get_synth().warmup()
    return {"status": "ok"}


@app.post("/synthesize", response_model=SynthesizeResponse)
def synthesize(req: SynthesizeRequest):
    logger.info(f"[design] synthesize: {req.text[:40]!r} desc={req.voice_description!r}")
    return get_synth().synthesize(req)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.workers.tts_design.main:app", host="0.0.0.0", port=PORT, reload=False)
