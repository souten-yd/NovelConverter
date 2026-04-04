"""Synthesizer implementations for TTS Base worker (Voice Clone mode).

MockSynthesizer: always available, generates silence/tone WAV.
Qwen3BaseSynthesizer: real Qwen3-TTS-12Hz-1.7B-Base for voice cloning.

Requires:
  - Qwen/Qwen3-TTS-Tokenizer-12Hz (shared tokenizer)
  - Qwen/Qwen3-TTS-12Hz-1.7B-Base (clone model)
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

TEMP_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent.parent.parent.parent / "data"))) / "temp"

# Default model paths – overridden by env vars or model_manager paths
_DEFAULT_BASE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
_DEFAULT_TOKENIZER = "Qwen/Qwen3-TTS-Tokenizer-12Hz"


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


# ── Real Qwen3-TTS Base (Voice Clone) ───────────────────────────────────────

class Qwen3BaseSynthesizer:
    """Real Qwen3-TTS voice clone synthesizer using 1.7B-Base model."""

    MODEL_ID = os.environ.get("TTS_BASE_MODEL_PATH", _DEFAULT_BASE_MODEL)
    TOKENIZER_ID = os.environ.get("TTS_TOKENIZER_PATH", _DEFAULT_TOKENIZER)

    def __init__(self):
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._load_error: Optional[str] = None
        self._load()

    def _resolve_model_path(self) -> str:
        """Resolve model path: check local model_manager path first, then HF ID."""
        try:
            from app.shared.model_manager import get_model_path, is_model_complete
            if is_model_complete("base"):
                return str(get_model_path("base"))
        except Exception:
            pass
        return self.MODEL_ID

    def _resolve_tokenizer_path(self) -> str:
        try:
            from app.shared.model_manager import get_model_path, is_model_complete
            if is_model_complete("tokenizer"):
                return str(get_model_path("tokenizer"))
        except Exception:
            pass
        return self.TOKENIZER_ID

    def _load(self):
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
            import torch

            model_path = self._resolve_model_path()
            tokenizer_path = self._resolve_tokenizer_path()

            logger.info(f"Loading Qwen3-TTS Base model from {model_path}")
            logger.info(f"Loading Qwen3-TTS tokenizer from {tokenizer_path}")

            self._processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True,
            )
            self._load_error = None
            logger.info("Qwen3-TTS Base loaded OK")
        except Exception as e:
            self._load_error = str(e)
            logger.error(f"Failed to load Qwen3-TTS Base: {e}")
            self._model = None

    def is_loaded(self) -> bool:
        return self._model is not None

    def get_load_error(self) -> Optional[str]:
        return self._load_error

    def available_models(self) -> list:
        return [self.MODEL_ID]

    def warmup(self) -> None:
        if not self._model:
            self._load()

    def synthesize(self, req: SynthesizeRequest) -> SynthesizeResponse:
        if not self._model:
            return SynthesizeResponse(
                success=False,
                error=f"Model not loaded: {self._load_error or 'unknown error'}",
                worker_id="qwen3-base",
            )

        # Validate reference audio for clone mode
        if not req.reference_audio_path:
            return SynthesizeResponse(
                success=False,
                error="Voice clone requires reference_audio_path",
                worker_id="qwen3-base",
            )

        ref_path = Path(req.reference_audio_path)
        if not ref_path.exists():
            return SynthesizeResponse(
                success=False,
                error=f"Reference audio not found: {req.reference_audio_path}",
                worker_id="qwen3-base",
            )

        if not req.reference_text:
            logger.warning("Voice clone: ref_text is empty; quality may be degraded")

        out_path = _resolve_output_path(req.output_path, "base")

        try:
            import torch
            import soundfile as sf
            import numpy as np
            from app.shared.audio_validator import validate_audio_array

            inputs = self._processor(
                text=req.text,
                ref_audio=req.reference_audio_path,
                ref_text=req.reference_text or "",
                return_tensors="pt",
            ).to(self._model.device)

            with torch.no_grad():
                output = self._model.generate(**inputs)

            audio_np = output.cpu().numpy().squeeze()

            # Validate before writing
            validation = validate_audio_array(audio_np, sample_rate=24000)
            if not validation.valid:
                logger.error(
                    f"Audio validation failed: {validation.failure_reason} "
                    f"(rms={validation.rms:.2e}, duration={validation.duration:.2f}s, "
                    f"model={self.MODEL_ID})"
                )
                return SynthesizeResponse(
                    success=False,
                    error=f"Generated audio failed validation: {validation.failure_reason}",
                    worker_id="qwen3-base",
                )

            sf.write(str(out_path), audio_np, samplerate=24000)

            return SynthesizeResponse(
                success=True,
                output_path=str(out_path),
                duration_seconds=validation.duration,
                worker_id="qwen3-base",
            )
        except Exception as e:
            logger.error(f"Qwen3BaseSynthesizer error: {e}")
            # Clean up partial output
            if out_path.exists():
                try:
                    out_path.unlink()
                except OSError:
                    pass
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
