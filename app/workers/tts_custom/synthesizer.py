"""Synthesizer for TTS Custom worker (Named Speaker mode)."""
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
    """Real Qwen3-TTS CustomVoice synthesizer using 1.7B-CustomVoice model."""

    MODEL_ID = os.environ.get("TTS_CUSTOM_MODEL_PATH", _DEFAULT_CUSTOM_MODEL)

    def __init__(self):
        self._model = None
        self._processor = None
        self._supported_speakers: List[str] = list(_BUILT_IN_SPEAKERS)
        self._supported_languages: List[str] = ["ja", "en", "zh"]
        self._load_error: Optional[str] = None
        self._device = "cpu"
        self._model_dir = self.MODEL_ID
        self._used_cpu_fallback = False
        self._cpu_fallback_reason: Optional[str] = None
        self._load()

    def _resolve_model_path(self) -> str:
        try:
            from app.shared.model_manager import get_model_path, is_model_complete
            if is_model_complete("custom"):
                return str(get_model_path("custom"))
        except Exception:
            pass
        return self.MODEL_ID

    def _load(self):
        try:
            import sys
            import transformers
            import qwen_tts
            import torch
            from transformers import AutoProcessor
            from qwen_tts import Qwen3TTSModel

            self._model_dir = self._resolve_model_path()
            selected_device = "cuda" if torch.cuda.is_available() else "cpu"
            self._used_cpu_fallback = False
            self._cpu_fallback_reason = None

            logger.info(f"sys.executable={sys.executable}")
            logger.info(f"transformers.__version__={transformers.__version__}")
            logger.info(f"qwen_tts.__file__={qwen_tts.__file__}")
            logger.info(f"torch.__version__={torch.__version__}")
            logger.info(f"torch.version.cuda={torch.version.cuda}")
            logger.info(f"torch.cuda.is_available()={torch.cuda.is_available()}")
            logger.info(f"selected_device={selected_device}")
            logger.info(f"model_dir={self._model_dir}")

            if selected_device == "cuda" and torch.version.cuda is None:
                self._used_cpu_fallback = True
                self._cpu_fallback_reason = "torch.cuda.is_available() is True but torch.version.cuda is None"
                selected_device = "cpu"
                logger.error(
                    "CUDA runtime inconsistency detected (%s); forcing CPU fallback.",
                    self._cpu_fallback_reason,
                )

            logger.info(f"Loading Qwen3-TTS Custom from {self._model_dir}")
            self._processor = AutoProcessor.from_pretrained(
                self._model_dir,
                trust_remote_code=True,
                fix_mistral_regex=True,
            )
            self._model = Qwen3TTSModel.from_pretrained(
                self._model_dir,
                trust_remote_code=True,
            )

            if hasattr(self._model, "to"):
                try:
                    self._model = self._model.to(selected_device)
                except Exception as move_error:
                    if selected_device == "cuda":
                        self._used_cpu_fallback = True
                        self._cpu_fallback_reason = f"CUDA move failed: {move_error}"
                        logger.error(
                            "Failed to move model to CUDA (%s); falling back to CPU.",
                            move_error,
                        )
                        self._model = self._model.to("cpu")
                        selected_device = "cpu"
                    else:
                        raise

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

            self._device = selected_device
            self._load_error = None
            logger.info(
                "Qwen3-TTS Custom loaded OK (device=%s, cpu_fallback=%s)",
                self._device,
                self._used_cpu_fallback,
            )
        except Exception as e:
            self._load_error = str(e)
            logger.exception("Failed to load Qwen3-TTS Custom")
            self._model = None
            self._processor = None
            self._device = "cpu"

    def is_loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    def get_load_error(self) -> Optional[str]:
        return self._load_error

    def get_runtime_status(self) -> dict:
        return {
            "device": self._device,
            "model_dir": self._model_dir,
            "cpu_fallback": self._used_cpu_fallback,
            "cpu_fallback_reason": self._cpu_fallback_reason,
        }

    def available_models(self) -> list:
        return [self.MODEL_ID]

    def available_speakers(self) -> list:
        return self._supported_speakers

    def warmup(self) -> None:
        if not self.is_loaded():
            self._load()

    def synthesize(self, req: SynthesizeRequest) -> SynthesizeResponse:
        if not self.is_loaded():
            return SynthesizeResponse(
                success=False,
                error=f"Model not loaded: {self._load_error or 'unknown error'}",
                worker_id="qwen3-custom",
            )

        out_path = _resolve_output_path(req.output_path, "custom")

        try:
            import soundfile as sf
            import torch
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

            inputs = self._processor(**processor_kwargs).to(self._device)

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
            logger.exception("Qwen3CustomSynthesizer error")
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
