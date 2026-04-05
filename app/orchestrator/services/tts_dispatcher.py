"""TTS dispatcher: routes synthesis requests to the appropriate worker."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import requests

from app.shared.logger import get_logger
from app.shared.schemas import SynthesizeRequest, SynthesizeResponse
from app.orchestrator.services.resource_manager import acquire_lease, release_lease

logger = get_logger("tts_dispatcher")

# ── Worker URLs (configurable via env) ────────────────────────────────────────

WORKER_URLS = {
    "base":   os.environ.get("TTS_BASE_URL",   "http://localhost:8001"),
    "custom": os.environ.get("TTS_CUSTOM_URL", "http://localhost:8002"),
    "design": os.environ.get("TTS_DESIGN_URL", "http://localhost:8003"),
}

MAX_RETRIES = 3
RETRY_DELAY = 2.0  # seconds


def get_worker_url(worker_type: str) -> str:
    return WORKER_URLS.get(worker_type, WORKER_URLS["custom"])


def is_worker_alive(worker_type: str) -> bool:
    url = get_worker_url(worker_type)
    try:
        r = requests.get(f"{url}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def synthesize(
    text: str,
    worker_type: str,
    output_path: Path,
    *,
    language: str = "ja",
    speaker: Optional[str] = None,
    instruct: Optional[str] = None,
    voice_description: Optional[str] = None,
    reference_audio_path: Optional[str] = None,
    reference_text: Optional[str] = None,
    speed: float = 1.0,
) -> SynthesizeResponse:
    """Dispatch a synthesis request with retry logic."""
    mode_map = {"base": "clone", "custom": "custom", "design": "design"}
    mode = mode_map.get(worker_type, "custom")

    req = SynthesizeRequest(
        text=text,
        mode=mode,
        language=language,
        speaker=speaker,
        instruct=instruct,
        voice_description=voice_description,
        reference_audio_path=reference_audio_path,
        reference_text=reference_text,
        speed=speed,
        output_path=str(output_path),
    )

    url = get_worker_url(worker_type)
    last_error: str = ""

    lease_reason = f"tts_synthesize_{worker_type}"
    acquire_lease(
        f"tts_{worker_type}",
        reason=lease_reason,
        model_id=worker_type,
        options={"worker_type": worker_type},
    )
    try:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    f"{url}/synthesize",
                    json=req.model_dump(),
                    timeout=120,
                )
                resp.raise_for_status()
                result = SynthesizeResponse(**resp.json())
                if result.success:
                    logger.info(f"Synthesized segment → {output_path} via {worker_type}")
                    return result
                last_error = result.error or "unknown error"
                logger.warning(f"Worker returned failure (attempt {attempt}): {last_error}")
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Synthesis attempt {attempt} failed: {e}")

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
    finally:
        release_lease(f"tts_{worker_type}", reason=lease_reason)

    return SynthesizeResponse(success=False, error=last_error)
