"""Synthesizer for TTS Custom worker.

Mock: generates tones at different frequencies per speaker.
Real: Qwen3-TTS CustomVoice mode.
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

logger = get_logger("synth.custom")

TEMP_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent.parent.parent.parent / "data"))) / "temp"

# Speaker → pitch (Hz) mapping for mock
_SPEAKER_PITCHES = {
    "narrator": 180.0,
    "Narrator": 180.0,
    "Aria": 280.0,
    "aria": 280.0,
    "default": 220.0,
}

_BUILT_IN_SPEAKERS = [
    "Aria", "Roger", "Sarah", "Laura", "Charlie",
    "George", "Callum", "River", "Liam", "Charlotte",
    "Alice", "Matilda", "Will", "Jessica", "Eric", "Chris", "Brian",
]


class MockSynthesizer:
    def __init__(self, worker_id: str = "mock_custom"):
        self.worker_id = worker_id
        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded

    def available_models(self) -> list:
        return ["mock-custom-v1"]

    def available_speakers(self) -> list:
        return _BUILT_IN_SPEAKERS

    def warmup(self) -> None:
        logger.info("MockSynthesizer (custom) warmup (no-op)")

    def synthesize(self, req: SynthesizeRequest) -> SynthesizeResponse:
        out_path = _resolve_output_path(req.output_path, "custom")
        try:
            freq = _SPEAKER_PITCHES.get(req.speaker or "default", 220.0)
            _write_tone_wav(out_path, req.text, frequency=freq, speed=req.speed)
            return SynthesizeResponse(
                success=True,
                output_path=str(out_path),
                duration_seconds=_estimate_duration(req.text, req.speed),
                worker_id=self.worker_id,
            )
        except Exception as e:
            logger.error(f"MockSynthesizer (custom) error: {e}")
            return SynthesizeResponse(success=False, error=str(e), worker_id=self.worker_id)


class Qwen3CustomSynthesizer:
    """Real Qwen3-TTS CustomVoice synthesizer.

    Uses the Qwen3-TTS model with a named speaker and optional instruct prompt.
    """

    MODEL_ID = os.environ.get("TTS_CUSTOM_MODEL_PATH", "Qwen/Qwen3-TTS")

    def __init__(self):
        self._model = None
        self._processor = None
        self._load()

    def _load(self):
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore
            import torch  # type: ignore

            logger.info(f"Loading Qwen3-TTS Custom from {self.MODEL_ID}")
            self._processor = AutoProcessor.from_pretrained(self.MODEL_ID, trust_remote_code=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_ID,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True,
            )
            logger.info("Qwen3-TTS Custom loaded OK")
        except Exception as e:
            logger.error(f"Failed to load Qwen3-TTS Custom: {e}")
            self._model = None

    def is_loaded(self) -> bool:
        return self._model is not None

    def available_models(self) -> list:
        return [self.MODEL_ID]

    def available_speakers(self) -> list:
        return _BUILT_IN_SPEAKERS

    def warmup(self) -> None:
        if not self._model:
            self._load()

    def synthesize(self, req: SynthesizeRequest) -> SynthesizeResponse:
        if not self._model:
            return SynthesizeResponse(success=False, error="Model not loaded", worker_id="qwen3-custom")

        out_path = _resolve_output_path(req.output_path, "custom")

        try:
            import torch
            import soundfile as sf  # type: ignore
            import numpy as np

            # Build prompt with speaker and instruct
            speaker = req.speaker or "Aria"
            instruct = req.instruct or ""
            prompt_text = req.text
            if instruct:
                prompt_text = f"[{instruct}] {prompt_text}"

            inputs = self._processor(
                text=prompt_text,
                speaker=speaker,
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
                worker_id="qwen3-custom",
            )
        except Exception as e:
            logger.error(f"Qwen3CustomSynthesizer error: {e}")
            return SynthesizeResponse(success=False, error=str(e), worker_id="qwen3-custom")


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
    return (max(len(text), 1) / 5.0) / max(speed, 0.1)


def _write_tone_wav(path: Path, text: str, frequency: float = 220.0, speed: float = 1.0):
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
