"""Synthesizer implementations for TTS Base worker.

MockSynthesizer: always available, generates silence/tone WAV.
Qwen3BaseSynthesizer: real Qwen3-TTS, loaded when USE_REAL_MODEL=true.
"""
from __future__ import annotations

import math
import os
import struct
import wave
from pathlib import Path
from typing import Optional

from app.shared.schemas import SynthesizeRequest, SynthesizeResponse
from app.shared.logger import get_logger

logger = get_logger("synth.base")

TEMP_DIR = Path(__file__).parent.parent.parent.parent / "data" / "temp"


# ── Mock ──────────────────────────────────────────────────────────────────────

class MockSynthesizer:
    """Generates a short tone WAV to simulate TTS output."""

    def __init__(self, worker_id: str = "mock_base"):
        self.worker_id = worker_id
        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded

    def available_models(self) -> list:
        return ["mock-base-v1"]

    def warmup(self) -> None:
        logger.info("MockSynthesizer warmup (no-op)")

    def synthesize(self, req: SynthesizeRequest) -> SynthesizeResponse:
        out_path = _resolve_output_path(req.output_path, "base")
        try:
            _write_tone_wav(out_path, text=req.text, frequency=220.0, speed=req.speed)
            return SynthesizeResponse(
                success=True,
                output_path=str(out_path),
                duration_seconds=_estimate_duration(req.text, req.speed),
                worker_id=self.worker_id,
            )
        except Exception as e:
            logger.error(f"MockSynthesizer error: {e}")
            return SynthesizeResponse(success=False, error=str(e), worker_id=self.worker_id)


# ── Real Qwen3-TTS Base ───────────────────────────────────────────────────────

class Qwen3BaseSynthesizer:
    """Real Qwen3-TTS voice clone synthesizer.

    Requirements (installed in .venv_tts_base):
        pip install qwen3-tts  # or the actual package
        soundfile numpy

    Set environment variables:
        TTS_BASE_MODEL_PATH – path to model weights (default: Qwen/Qwen3-TTS)
    """

    MODEL_ID = os.environ.get("TTS_BASE_MODEL_PATH", "Qwen/Qwen3-TTS")

    def __init__(self):
        self._model = None
        self._processor = None
        self._load()

    def _load(self):
        try:
            # Import here so import errors don't break mock mode
            from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore
            import torch  # type: ignore

            logger.info(f"Loading Qwen3-TTS Base from {self.MODEL_ID}")
            self._processor = AutoProcessor.from_pretrained(self.MODEL_ID, trust_remote_code=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_ID,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True,
            )
            logger.info("Qwen3-TTS Base loaded OK")
        except Exception as e:
            logger.error(f"Failed to load Qwen3-TTS Base: {e}")
            self._model = None

    def is_loaded(self) -> bool:
        return self._model is not None

    def available_models(self) -> list:
        return [self.MODEL_ID]

    def warmup(self) -> None:
        if not self._model:
            self._load()

    def synthesize(self, req: SynthesizeRequest) -> SynthesizeResponse:
        if not self._model:
            return SynthesizeResponse(success=False, error="Model not loaded", worker_id="qwen3-base")

        out_path = _resolve_output_path(req.output_path, "base")

        try:
            import torch
            import soundfile as sf  # type: ignore
            import numpy as np

            # Build inputs for voice clone
            inputs = self._processor(
                text=req.text,
                reference_audio=req.reference_audio_path,
                reference_text=req.reference_text,
                return_tensors="pt",
            ).to(self._model.device)

            with torch.no_grad():
                output = self._model.generate(**inputs)

            audio_np = output.cpu().numpy().squeeze()
            sf.write(str(out_path), audio_np, samplerate=24000)

            duration = len(audio_np) / 24000.0
            return SynthesizeResponse(
                success=True,
                output_path=str(out_path),
                duration_seconds=duration,
                worker_id="qwen3-base",
            )
        except Exception as e:
            logger.error(f"Qwen3BaseSynthesizer error: {e}")
            return SynthesizeResponse(success=False, error=str(e), worker_id="qwen3-base")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_output_path(requested: Optional[str], prefix: str) -> Path:
    if requested:
        p = Path(requested)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    import uuid
    return TEMP_DIR / f"{prefix}_{uuid.uuid4().hex[:8]}.wav"


def _estimate_duration(text: str, speed: float = 1.0) -> float:
    """Rough estimate: ~5 chars/second for Japanese."""
    chars = max(len(text), 1)
    return (chars / 5.0) / max(speed, 0.1)


def _write_tone_wav(path: Path, text: str, frequency: float = 220.0, speed: float = 1.0):
    """Write a sine-wave WAV. Duration estimated from text length."""
    sample_rate = 24000
    duration = _estimate_duration(text, speed)
    n_samples = int(sample_rate * duration)
    amplitude = 16000

    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        samples = []
        for i in range(n_samples):
            t = i / sample_rate
            val = int(amplitude * math.sin(2 * math.pi * frequency * t))
            # gentle fade in/out
            if i < 1000:
                val = int(val * i / 1000)
            elif i > n_samples - 1000:
                val = int(val * (n_samples - i) / 1000)
            samples.append(struct.pack("<h", val))
        wf.writeframes(b"".join(samples))
