"""Synthesizer for TTS Custom worker (Named Speaker mode).

MockSynthesizer: generates tones at different frequencies per speaker.
Qwen3CustomSynthesizer: real Qwen3-TTS-12Hz-1.7B-CustomVoice model.

Requires:
  - Qwen/Qwen3-TTS-Tokenizer-12Hz (shared tokenizer)
  - Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice (custom voice model)
"""
from __future__ import annotations

import math
import os
import struct
import wave
from pathlib import Path
from typing import List, Optional

from app.shared.schemas import SynthesizeRequest, SynthesizeResponse
from app.shared.logger import get_logger

logger = get_logger("synth.custom")

TEMP_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent.parent.parent.parent / "data"))) / "temp"

_DEFAULT_CUSTOM_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
_DEFAULT_TOKENIZER = "Qwen/Qwen3-TTS-Tokenizer-12Hz"

# Speaker → pitch (Hz) mapping for mock
_SPEAKER_PITCHES = {
    "narrator": 180.0,
    "Narrator": 180.0,
    "Aria": 280.0,
    "aria": 280.0,
    "default": 220.0,
}

# Fallback built-in speakers (used if model doesn't expose get_supported_speakers)
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
    """Real Qwen3-TTS CustomVoice synthesizer using 1.7B-CustomVoice model.

    Uses named speakers (Aria, Roger, etc.) with optional instruct prompts.
    """

    MODEL_ID = os.environ.get("TTS_CUSTOM_MODEL_PATH", _DEFAULT_CUSTOM_MODEL)
    TOKENIZER_ID = os.environ.get("TTS_TOKENIZER_PATH", _DEFAULT_TOKENIZER)

    def __init__(self):
        self._model = None
        self._processor = None
        self._supported_speakers: List[str] = list(_BUILT_IN_SPEAKERS)
        self._supported_languages: List[str] = ["ja", "en", "zh"]
        self._load_error: Optional[str] = None
        self._load()

    def _resolve_model_path(self) -> str:
        try:
            from app.shared.model_manager import get_model_path, is_model_complete
            if is_model_complete("custom"):
                return str(get_model_path("custom"))
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
            from transformers import AutoModelForCausalLM, AutoProcessor
            import torch

            model_path = self._resolve_model_path()
            tokenizer_path = self._resolve_tokenizer_path()

            logger.info(f"Loading Qwen3-TTS Custom from {model_path}")
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

            # Try to get supported speakers/languages from model
            if hasattr(self._model, "get_supported_speakers"):
                try:
                    self._supported_speakers = list(self._model.get_supported_speakers())
                    logger.info(f"Model speakers: {self._supported_speakers}")
                except Exception as e:
                    logger.warning(f"Could not get speakers from model: {e}")

            if hasattr(self._model, "get_supported_languages"):
                try:
                    self._supported_languages = list(self._model.get_supported_languages())
                    logger.info(f"Model languages: {self._supported_languages}")
                except Exception as e:
                    logger.warning(f"Could not get languages from model: {e}")

            self._load_error = None
            logger.info("Qwen3-TTS Custom loaded OK")
        except Exception as e:
            self._load_error = str(e)
            logger.error(f"Failed to load Qwen3-TTS Custom: {e}")
            self._model = None

    def is_loaded(self) -> bool:
        return self._model is not None

    def get_load_error(self) -> Optional[str]:
        return self._load_error

    def available_models(self) -> list:
        return [self.MODEL_ID]

    def available_speakers(self) -> list:
        return self._supported_speakers

    def warmup(self) -> None:
        if not self._model:
            self._load()

    def synthesize(self, req: SynthesizeRequest) -> SynthesizeResponse:
        if not self._model:
            return SynthesizeResponse(
                success=False,
                error=f"Model not loaded: {self._load_error or 'unknown error'}",
                worker_id="qwen3-custom",
            )

        out_path = _resolve_output_path(req.output_path, "custom")

        try:
            import torch
            import soundfile as sf
            import numpy as np
            from app.shared.audio_validator import validate_audio_array

            speaker = req.speaker or "Aria"

            # Warn if speaker not in supported list
            if speaker not in self._supported_speakers:
                logger.warning(
                    f"Speaker '{speaker}' not in supported list: {self._supported_speakers}. "
                    f"Falling back to 'Aria'."
                )
                speaker = "Aria"

            # Warn if language not supported
            lang = req.language or "ja"
            if lang not in self._supported_languages:
                logger.warning(f"Language '{lang}' may not be supported. Supported: {self._supported_languages}")

            # Build processor inputs
            processor_kwargs = {
                "text": req.text,
                "speaker": speaker,
                "return_tensors": "pt",
            }

            # Add instruct if provided
            if req.instruct:
                processor_kwargs["instruct"] = req.instruct

            inputs = self._processor(**processor_kwargs).to(self._model.device)

            with torch.no_grad():
                output = self._model.generate(**inputs)

            audio_np = output.cpu().numpy().squeeze()

            # Validate before writing
            validation = validate_audio_array(audio_np, sample_rate=24000)
            if not validation.valid:
                logger.error(
                    f"Audio validation failed: {validation.failure_reason} "
                    f"(speaker={speaker}, rms={validation.rms:.2e}, "
                    f"duration={validation.duration:.2f}s)"
                )
                return SynthesizeResponse(
                    success=False,
                    error=f"Generated audio failed validation: {validation.failure_reason}",
                    worker_id="qwen3-custom",
                )

            sf.write(str(out_path), audio_np, samplerate=24000)

            return SynthesizeResponse(
                success=True,
                output_path=str(out_path),
                duration_seconds=validation.duration,
                worker_id="qwen3-custom",
            )
        except Exception as e:
            logger.error(f"Qwen3CustomSynthesizer error: {e}")
            if out_path.exists():
                try:
                    out_path.unlink()
                except OSError:
                    pass
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
