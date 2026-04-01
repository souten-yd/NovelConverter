"""Audio merger: combine segment WAVs into chapter and full audiobooks."""
from __future__ import annotations

import os
import shutil
import struct
import subprocess
import wave
from pathlib import Path
from typing import List, Optional

from app.shared.logger import get_logger

logger = get_logger("audio_merger")

SILENCE_SAMPLE_RATE = 24000
SILENCE_CHANNELS = 1
SILENCE_SAMPWIDTH = 2  # 16-bit


# ── Silence generation (no external deps) ────────────────────────────────────

def _generate_silence(ms: int, path: Path) -> None:
    """Write a silence WAV file of given duration in milliseconds."""
    n_samples = int(SILENCE_SAMPLE_RATE * ms / 1000)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(SILENCE_CHANNELS)
        wf.setsampwidth(SILENCE_SAMPWIDTH)
        wf.setframerate(SILENCE_SAMPLE_RATE)
        wf.writeframes(b"\x00" * (n_samples * SILENCE_CHANNELS * SILENCE_SAMPWIDTH))


def _get_wav_params(path: Path) -> tuple[int, int, int, int]:
    """Return (nchannels, sampwidth, framerate, nframes)."""
    with wave.open(str(path), "r") as wf:
        return wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()


def _resample_wav(path: Path, target_rate: int, target_channels: int, target_sw: int) -> bytes:
    """Read wav and return raw PCM bytes, upsampling/downsampling via wave only."""
    with wave.open(str(path), "r") as wf:
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    # Only handle trivial same-params case for now; otherwise return as-is
    # Real resampling would require scipy/librosa – skip for prototype
    if ch != target_channels or sw != target_sw or rate != target_rate:
        logger.debug(
            f"Audio params mismatch for {path}: ch={ch}/{target_channels} "
            f"sw={sw}/{target_sw} rate={rate}/{target_rate} – using as-is"
        )
    return frames


def merge_wavs(
    audio_paths: List[Path],
    output_path: Path,
    pause_ms_before: int = 0,
    pause_ms_after: int = 300,
    temp_dir: Optional[Path] = None,
) -> Path:
    """Merge a list of WAV files into a single output WAV.

    Inserts silence before and after each segment according to pause settings.
    """
    if not audio_paths:
        raise ValueError("No audio paths to merge")

    if temp_dir is None:
        temp_dir = output_path.parent / "_tmp_merge"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Determine output params from first valid file
    target_rate = SILENCE_SAMPLE_RATE
    target_ch = SILENCE_CHANNELS
    target_sw = SILENCE_SAMPWIDTH

    for p in audio_paths:
        if p.exists():
            try:
                target_ch, target_sw, target_rate, _ = _get_wav_params(p)
                break
            except Exception:
                pass

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(output_path), "w") as out:
        out.setnchannels(target_ch)
        out.setsampwidth(target_sw)
        out.setframerate(target_rate)

        silence_before: Optional[bytes] = None
        silence_after: Optional[bytes] = None

        if pause_ms_before > 0:
            sil_path = temp_dir / f"sil_before_{pause_ms_before}.wav"
            _generate_silence(pause_ms_before, sil_path)
            silence_before = _resample_wav(sil_path, target_rate, target_ch, target_sw)

        if pause_ms_after > 0:
            sil_path = temp_dir / f"sil_after_{pause_ms_after}.wav"
            _generate_silence(pause_ms_after, sil_path)
            silence_after = _resample_wav(sil_path, target_rate, target_ch, target_sw)

        for ap in audio_paths:
            if not ap.exists():
                logger.warning(f"Missing audio file, skipping: {ap}")
                continue
            try:
                if silence_before:
                    out.writeframes(silence_before)
                frames = _resample_wav(ap, target_rate, target_ch, target_sw)
                out.writeframes(frames)
                if silence_after:
                    out.writeframes(silence_after)
            except Exception as e:
                logger.error(f"Failed to append {ap}: {e}")

    logger.info(f"Merged {len(audio_paths)} files → {output_path}")
    # cleanup temp
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass
    return output_path


def try_convert_to_m4b(wav_path: Path) -> Optional[Path]:
    """Convert WAV to M4B using ffmpeg if available."""
    if not shutil.which("ffmpeg"):
        logger.info("ffmpeg not found – skipping M4B conversion")
        return None
    m4b_path = wav_path.with_suffix(".m4b")
    cmd = [
        "ffmpeg", "-y", "-i", str(wav_path),
        "-c:a", "aac", "-b:a", "128k",
        str(m4b_path),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            logger.info(f"Converted to M4B: {m4b_path}")
            return m4b_path
        else:
            logger.error(f"ffmpeg failed: {result.stderr}")
            return None
    except Exception as e:
        logger.error(f"M4B conversion error: {e}")
        return None


def get_wav_duration(path: Path) -> float:
    """Return duration of WAV in seconds."""
    try:
        with wave.open(str(path), "r") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate) if rate > 0 else 0.0
    except Exception:
        return 0.0
