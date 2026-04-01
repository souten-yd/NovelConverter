"""Synthesizer for TTS Design worker.

Mock: generates tones varied by voice_description hash.
Real: Qwen3-TTS VoiceDesign mode.
"""
from __future__ import annotations

import hashlib
import math
import os
import struct
import wave
from pathlib import Path
from typing import Optional

from app.shared.schemas import SynthesizeRequest, SynthesizeResponse
from app.shared.logger import get_logger

logger = get_logger("synth.design")

TEMP_DIR = Path(__file__).parent.parent.parent.parent / "data" / "temp"


class MockSynthesizer:
    def __init__(self, worker_id: str = "mock_design"):
        self.worker_id = worker_id
        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded

    def available_models(self) -> list:
        return ["mock-design-v1"]

    def warmup(self) -> None:
        logger.info("MockSynthesizer (design) warmup (no-op)")

    def synthesize(self, req: SynthesizeRequest) -> SynthesizeResponse:
        out_path = _resolve_output_path(req.output_path, "design")
        try:
            # vary frequency based on voice_description hash
            desc = req.voice_description or "default"
            h = int(hashlib.md5(desc.encode()).hexdigest()[:4], 16)
            freq = 150.0 + (h % 200)
            _write_tone_wav(out_path, req.text, frequency=freq, speed=req.speed)
            return SynthesizeResponse(
                success=True,
                output_path=str(out_path),
                duration_seconds=_estimate_duration(req.text, req.speed),
                worker_id=self.worker_id,
            )
        except Exception as e:
            logger.error(f"MockSynthesizer (design) error: {e}")
            return SynthesizeResponse(success=False, error=str(e), worker_id=self.worker_id)


class Qwen3DesignSynthesizer:
    """Real Qwen3-TTS VoiceDesign synthesizer.

    Uses natural language voice_description to generate speech.
    """

    MODEL_ID = os.environ.get("TTS_DESIGN_MODEL_PATH", "Qwen/Qwen3-TTS")

    def __init__(self):
        self._model = None
        self._processor = None
        self._load()

    def _load(self):
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore
            import torch  # type: ignore

            logger.info(f"Loading Qwen3-TTS Design from {self.MODEL_ID}")
            self._processor = AutoProcessor.from_pretrained(self.MODEL_ID, trust_remote_code=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_ID,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True,
            )
            logger.info("Qwen3-TTS Design loaded OK")
        except Exception as e:
            logger.error(f"Failed to load Qwen3-TTS Design: {e}")
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
            return SynthesizeResponse(success=False, error="Model not loaded", worker_id="qwen3-design")

        out_path = _resolve_output_path(req.output_path, "design")

        try:
            import torch
            import soundfile as sf  # type: ignore
            import numpy as np

            voice_description = req.voice_description or "A pleasant neutral voice"
            inputs = self._processor(
                text=req.text,
                voice_description=voice_description,
                language=req.language or "ja",
                return_tensors="pt",
            ).to(self._model.device)

            with torch.no_grad():
                output = self._model.generate(**inputs)

            audio_np = output.cpu().numpy().squeeze()
            sf.write(str(out_path), audio_np, samplerate=24000)

            return SynthesizeResponse(
                success=True,
                output_path=str(out_path),
                duration_seconds=len(audio_np) / 24000.0,
                worker_id="qwen3-design",
            )
        except Exception as e:
            logger.error(f"Qwen3DesignSynthesizer error: {e}")
            return SynthesizeResponse(success=False, error=str(e), worker_id="qwen3-design")


def _resolve_output_path(requested: Optional[str], prefix: str) -> Path:
    if requested:
        p = Path(requested)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    import uuid
    return TEMP_DIR / f"{prefix}_{uuid.uuid4().hex[:8]}.wav"


def _estimate_duration(text: str, speed: float = 1.0) -> float:
    return (max(len(text), 1) / 5.0) / max(speed, 0.1)


def _write_tone_wav(path: Path, text: str, frequency: float = 260.0, speed: float = 1.0):
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
            if i < 1000:
                val = int(val * i / 1000)
            elif i > n_samples - 1000:
                val = int(val * (n_samples - i) / 1000)
            samples.append(struct.pack("<h", val))
        wf.writeframes(b"".join(samples))
