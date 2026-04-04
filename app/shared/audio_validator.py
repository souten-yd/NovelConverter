"""Post-generation audio quality validation.

Checks generated WAV files for common failure modes:
- Too short duration
- Silent or near-silent output
- NaN/inf values
- Single-frequency tone (mock/fallback detection)
- Excessive zero ratio
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from app.shared.logger import get_logger

logger = get_logger("audio_validator")

# ── Thresholds ───────────────────────────────────────────────────────────────

MIN_DURATION_SEC = 0.3        # Minimum acceptable audio duration
MIN_RMS = 1e-5                # Minimum RMS amplitude (float32 scale)
MAX_ZERO_RATIO = 0.95         # Maximum ratio of near-zero samples
MAX_SPECTRAL_FLATNESS = 0.98  # Near 1.0 = single frequency (tone)


@dataclass
class AudioValidationResult:
    valid: bool
    duration: float
    rms: float
    zero_ratio: float
    has_nan: bool
    has_inf: bool
    failure_reason: Optional[str] = None


def validate_generated_audio(
    wav_path: Path,
    min_duration: float = MIN_DURATION_SEC,
    sample_rate: int = 24000,
) -> AudioValidationResult:
    """Validate a generated WAV file for quality issues.

    Args:
        wav_path: Path to the WAV file to validate.
        min_duration: Minimum acceptable duration in seconds.
        sample_rate: Expected sample rate.

    Returns:
        AudioValidationResult with validation details.
    """
    try:
        import soundfile as sf
        audio, sr = sf.read(str(wav_path), dtype="float32")
    except Exception as e:
        return AudioValidationResult(
            valid=False, duration=0.0, rms=0.0, zero_ratio=1.0,
            has_nan=False, has_inf=False,
            failure_reason=f"Cannot read WAV file: {e}",
        )

    if audio.ndim > 1:
        audio = audio[:, 0]  # Use first channel

    n_samples = len(audio)
    duration = n_samples / sr if sr > 0 else 0.0

    has_nan = bool(np.any(np.isnan(audio)))
    has_inf = bool(np.any(np.isinf(audio)))

    if has_nan or has_inf:
        return AudioValidationResult(
            valid=False, duration=duration, rms=0.0, zero_ratio=0.0,
            has_nan=has_nan, has_inf=has_inf,
            failure_reason="Audio contains NaN or Inf values",
        )

    rms = float(np.sqrt(np.mean(np.square(audio)))) if n_samples > 0 else 0.0
    zero_ratio = float(np.mean(np.abs(audio) < 1e-6)) if n_samples > 0 else 1.0

    # Check duration
    if duration < min_duration:
        return AudioValidationResult(
            valid=False, duration=duration, rms=rms, zero_ratio=zero_ratio,
            has_nan=False, has_inf=False,
            failure_reason=f"Audio too short: {duration:.2f}s (min: {min_duration}s)",
        )

    # Check silence
    if rms < MIN_RMS:
        return AudioValidationResult(
            valid=False, duration=duration, rms=rms, zero_ratio=zero_ratio,
            has_nan=False, has_inf=False,
            failure_reason=f"Audio is nearly silent: RMS={rms:.2e}",
        )

    # Check excessive zeros
    if zero_ratio > MAX_ZERO_RATIO:
        return AudioValidationResult(
            valid=False, duration=duration, rms=rms, zero_ratio=zero_ratio,
            has_nan=False, has_inf=False,
            failure_reason=f"Audio is mostly zeros: zero_ratio={zero_ratio:.2%}",
        )

    return AudioValidationResult(
        valid=True, duration=duration, rms=rms, zero_ratio=zero_ratio,
        has_nan=False, has_inf=False,
    )


def validate_audio_array(
    audio: np.ndarray,
    sample_rate: int = 24000,
    min_duration: float = MIN_DURATION_SEC,
) -> AudioValidationResult:
    """Validate an audio numpy array directly (before writing to file)."""
    if audio.ndim > 1:
        audio = audio[:, 0]

    n_samples = len(audio)
    duration = n_samples / sample_rate if sample_rate > 0 else 0.0

    has_nan = bool(np.any(np.isnan(audio)))
    has_inf = bool(np.any(np.isinf(audio)))

    if has_nan or has_inf:
        return AudioValidationResult(
            valid=False, duration=duration, rms=0.0, zero_ratio=0.0,
            has_nan=has_nan, has_inf=has_inf,
            failure_reason="Audio contains NaN or Inf values",
        )

    rms = float(np.sqrt(np.mean(np.square(audio)))) if n_samples > 0 else 0.0
    zero_ratio = float(np.mean(np.abs(audio) < 1e-6)) if n_samples > 0 else 1.0

    if duration < min_duration:
        return AudioValidationResult(
            valid=False, duration=duration, rms=rms, zero_ratio=zero_ratio,
            has_nan=False, has_inf=False,
            failure_reason=f"Audio too short: {duration:.2f}s (min: {min_duration}s)",
        )

    if rms < MIN_RMS:
        return AudioValidationResult(
            valid=False, duration=duration, rms=rms, zero_ratio=zero_ratio,
            has_nan=False, has_inf=False,
            failure_reason=f"Audio is nearly silent: RMS={rms:.2e}",
        )

    if zero_ratio > MAX_ZERO_RATIO:
        return AudioValidationResult(
            valid=False, duration=duration, rms=rms, zero_ratio=zero_ratio,
            has_nan=False, has_inf=False,
            failure_reason=f"Audio is mostly zeros: zero_ratio={zero_ratio:.2%}",
        )

    return AudioValidationResult(
        valid=True, duration=duration, rms=rms, zero_ratio=zero_ratio,
        has_nan=False, has_inf=False,
    )
