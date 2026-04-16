"""Synthesizer for TTS Design worker (Voice Description mode)."""
from __future__ import annotations

import hashlib
import math
import os
import subprocess
import struct
import sys
import wave
from pathlib import Path
from typing import Optional

from app.shared.schemas import SynthesizeRequest, SynthesizeResponse
from app.shared.logger import get_logger
from app.shared.language_codes import normalize_tts_language

logger = get_logger("synth.design")

TEMP_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent.parent.parent.parent / "data"))) / "temp"

_DEFAULT_DESIGN_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"


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
    """Real Qwen3-TTS VoiceDesign synthesizer using 1.7B-VoiceDesign model."""

    MODEL_ID = os.environ.get("TTS_DESIGN_MODEL_PATH", _DEFAULT_DESIGN_MODEL)

    def __init__(self):
        self._model = None
        self._processor = None
        self._load_error: Optional[str] = None
        self._device = "cpu"
        self._model_dir = self.MODEL_ID
        self._used_cpu_fallback = False
        self._cpu_fallback_reason: Optional[str] = None
        self._backend_name = "qwen3-design"
        self._python_executable = sys.executable
        self._sys_path = list(sys.path)
        self._import_status: dict[str, str] = {}
        self._pip_show_qwen_tts = ""
        self._tokenizer_path = os.environ.get("TTS_TOKENIZER_PATH", "Qwen/Qwen3-TTS-Tokenizer-12Hz")
        self._dtype = "unknown"
        self._attention_backend = "unknown"
        self._load()

    def _resolve_model_path(self) -> str:
        try:
            from app.shared.model_manager import get_model_path, is_model_complete
            if is_model_complete("design"):
                return str(get_model_path("design"))
        except Exception:
            pass
        return self.MODEL_ID

    def _load(self):
        try:
            import transformers
            import qwen_tts
            import torch
            import torchaudio
            from transformers import AutoProcessor
            from qwen_tts import Qwen3TTSModel

            self._model_dir = self._resolve_model_path()
            selected_device = "cuda" if torch.cuda.is_available() else "cpu"
            self._used_cpu_fallback = False
            self._cpu_fallback_reason = None

            self._python_executable = sys.executable
            self._sys_path = list(sys.path)
            self._import_status = self._collect_import_status()
            self._pip_show_qwen_tts = self._collect_pip_show()
            self._dtype = "float16" if selected_device == "cuda" else "float32"
            self._attention_backend = "sdpa" if selected_device == "cuda" else "eager"

            logger.info(f"sys.executable={self._python_executable}")
            logger.info(f"sys.path={self._sys_path}")
            logger.info(f"import_status={self._import_status}")
            logger.info("pip show qwen-tts:\n%s", self._pip_show_qwen_tts or "(empty)")
            logger.info(f"transformers.__version__={transformers.__version__}")
            logger.info(f"qwen_tts.__file__={qwen_tts.__file__}")
            logger.info(f"torch.__version__={torch.__version__}")
            logger.info(f"torchaudio.__version__={torchaudio.__version__}")
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

            logger.info(f"Loading Qwen3-TTS Design from {self._model_dir}")
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

            self._device = selected_device
            self._load_error = None
            logger.info(
                "Qwen3-TTS Design loaded OK (device=%s, cpu_fallback=%s)",
                self._device,
                self._used_cpu_fallback,
            )
        except Exception as e:
            err_msg = str(e)
            if "No module named 'qwen_tts'" in err_msg:
                err_msg = (
                    "Qwen3-TTS backend import failed: No module named 'qwen_tts'. "
                    "The worker environment is missing the qwen-tts package."
                )
            self._load_error = err_msg
            logger.exception("Failed to load Qwen3-TTS Design")
            self._model = None
            self._processor = None
            self._device = "cpu"

    def is_loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    def get_load_error(self) -> Optional[str]:
        return self._load_error

    def get_runtime_status(self) -> dict:
        return {
            "backend_name": self._backend_name,
            "python_executable": self._python_executable,
            "sys_path": self._sys_path,
            "import_status": self._import_status,
            "pip_show_qwen_tts": self._pip_show_qwen_tts,
            "device": self._device,
            "model_dir": self._model_dir,
            "model_root_exists": Path(self._model_dir).exists() if "/" in self._model_dir else False,
            "tokenizer_path": self._tokenizer_path,
            "tokenizer_exists": Path(self._tokenizer_path).exists() if "/" in self._tokenizer_path else False,
            "selected_dtype": self._dtype,
            "selected_attention_backend": self._attention_backend,
            "cpu_fallback": self._used_cpu_fallback,
            "cpu_fallback_reason": self._cpu_fallback_reason,
        }

    def _collect_import_status(self) -> dict[str, str]:
        modules = ("qwen_tts", "transformers", "torch", "torchaudio")
        status: dict[str, str] = {}
        for module in modules:
            try:
                __import__(module)
                status[module] = "ok"
            except Exception as e:  # pragma: no cover - diagnostic only
                status[module] = f"ng: {e}"
        return status

    def _collect_pip_show(self) -> str:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", "qwen-tts"],
                check=False,
                capture_output=True,
                text=True,
                timeout=20,
            )
            output = (result.stdout or "") + (result.stderr or "")
            return output.strip()
        except Exception as e:  # pragma: no cover - diagnostic only
            return f"failed to run pip show: {e}"

    def available_models(self) -> list:
        return [self.MODEL_ID]

    def warmup(self) -> None:
        if not self.is_loaded():
            self._load()

    def synthesize(self, req: SynthesizeRequest) -> SynthesizeResponse:
        if not self.is_loaded():
            return SynthesizeResponse(
                success=False,
                error=self._load_error or "Model not loaded: unknown error",
                worker_id="qwen3-design",
            )

        out_path = _resolve_output_path(req.output_path, "design")

        # Validate voice description
        voice_description = (req.voice_description or "").strip()
        if not voice_description:
            logger.warning("Voice description is empty, using default")
            voice_description = "A clear, natural Japanese female voice with moderate pace"

        if len(voice_description) < 5:
            logger.warning(f"Voice description very short ({len(voice_description)} chars), may produce poor results")

        try:
            import soundfile as sf
            from app.shared.audio_validator import validate_audio_array

            if not hasattr(self._model, "generate_voice_design"):
                return SynthesizeResponse(
                    success=False,
                    error="Loaded qwen-tts model does not support generate_voice_design",
                    worker_id="qwen3-design",
                )

            output = self._model.generate_voice_design(
                text=req.text,
                language=normalize_tts_language(req.language),
                instruct=voice_description,
            )
            audio_np, sample_rate = _extract_first_audio_and_rate(output, default_sample_rate=24000)

            # Validate before writing
            validation = validate_audio_array(audio_np, sample_rate=sample_rate)
            if not validation.valid:
                logger.error(
                    f"Audio validation failed: {validation.failure_reason} "
                    f"(desc='{voice_description[:50]}', rms={validation.rms:.2e}, "
                    f"duration={validation.duration:.2f}s)"
                )
                return SynthesizeResponse(
                    success=False,
                    error=f"Generated audio failed validation: {validation.failure_reason}",
                    worker_id="qwen3-design",
                )

            sf.write(str(out_path), audio_np, samplerate=sample_rate)

            return SynthesizeResponse(
                success=True,
                output_path=str(out_path),
                duration_seconds=validation.duration,
                worker_id="qwen3-design",
            )
        except Exception as e:
            logger.exception("Qwen3DesignSynthesizer error")
            if out_path.exists():
                try:
                    out_path.unlink()
                except OSError:
                    pass
            return SynthesizeResponse(success=False, error=str(e), worker_id="qwen3-design")


def _resolve_output_path(requested: Optional[str], prefix: str) -> Path:
    if requested:
        p = Path(requested)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    import uuid
    return TEMP_DIR / f"{prefix}_{uuid.uuid4().hex[:8]}.wav"


def _extract_first_audio_and_rate(output, default_sample_rate: int = 24000):
    sample_rate = default_sample_rate
    wavs = output
    if isinstance(output, tuple) and len(output) == 2:
        wavs, sample_rate = output
    if isinstance(wavs, list):
        wavs = wavs[0] if wavs else []
    if hasattr(wavs, "detach"):
        wavs = wavs.detach()
    if hasattr(wavs, "cpu"):
        wavs = wavs.cpu()
    if hasattr(wavs, "numpy"):
        wavs = wavs.numpy()
    if hasattr(wavs, "squeeze"):
        wavs = wavs.squeeze()
    return wavs, sample_rate


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
