"""Voice mapping API + Voice Design Studio API."""
from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.shared.database import get_db
from app.shared.models import Segment, Speaker, VoiceProfile, VoicePreset
from app.shared.paths import get_data_dir
from app.shared.schemas import SpeakerOut, VoiceMappingBatch, VoiceProfileOut
from app.shared.logger import get_logger

logger = get_logger("router.voice_mapping")
router = APIRouter(prefix="/api/projects", tags=["voice_mapping"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_speaker_or_404(project_id: str, speaker_id: str, db: Session) -> Speaker:
    sp = db.get(Speaker, speaker_id)
    if not sp or sp.project_id != project_id:
        raise HTTPException(status_code=404, detail=f"Speaker {speaker_id} not found in project {project_id}")
    return sp


def _temp_dir() -> Path:
    d = get_data_dir() / "temp" / "voice_preview"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _resolve_profile_data_with_preset(profile_data: dict[str, Any], db: Session) -> dict[str, Any]:
    """Resolve voice profile synthesis fields from preset_id when selected.

    Backward compatibility: when preset_id is empty/None, keep manually entered
    fields exactly as provided.
    """
    resolved = dict(profile_data)
    preset_id = resolved.get("preset_id")
    if not preset_id:
        return resolved

    preset = db.get(VoicePreset, preset_id)
    if not preset:
        raise HTTPException(status_code=404, detail=f"Voice preset {preset_id} not found")

    synth = preset.synthesis_params or {}
    resolved["worker_type"] = preset.engine_type
    resolved["speaker_name"] = synth.get("speaker_name")
    resolved["instruct"] = synth.get("instruct")
    resolved["voice_description"] = synth.get("voice_description")
    resolved["reference_audio_path"] = synth.get("reference_audio_path")
    resolved["reference_text"] = synth.get("reference_text")
    if "language" in synth:
        resolved["language"] = synth["language"]
    if "speed" in synth:
        resolved["speed"] = synth["speed"]
    if "pause_ms_before" in synth:
        resolved["pause_ms_before"] = synth["pause_ms_before"]
    if "pause_ms_after" in synth:
        resolved["pause_ms_after"] = synth["pause_ms_after"]
    if "volume_gain_db" in synth:
        resolved["volume_gain_db"] = synth["volume_gain_db"]
    return resolved


# ── Existing voice mapping endpoints ─────────────────────────────────────────

@router.get("/{project_id}/speakers", response_model=list[SpeakerOut])
def list_speakers(project_id: str, db: Session = Depends(get_db)):
    return db.query(Speaker).filter(Speaker.project_id == project_id).all()


@router.post("/{project_id}/voice_mappings")
def set_voice_mappings(
    project_id: str,
    body: VoiceMappingBatch,
    db: Session = Depends(get_db),
):
    updated = 0
    for entry in body.mappings:
        sp = db.get(Speaker, entry.speaker_id)
        if not sp or sp.project_id != project_id:
            raise HTTPException(status_code=404, detail=f"Speaker {entry.speaker_id} not found")

        # upsert single voice profile per speaker
        vp = (
            db.query(VoiceProfile)
            .filter(VoiceProfile.speaker_id == entry.speaker_id)
            .first()
        )
        profile_data = _resolve_profile_data_with_preset(entry.profile.model_dump(), db)
        if vp:
            for k, v in profile_data.items():
                setattr(vp, k, v)
        else:
            vp = VoiceProfile(speaker_id=entry.speaker_id, **profile_data)
            db.add(vp)
        updated += 1

    db.commit()
    return {"updated": updated}


@router.get("/{project_id}/voice_profiles", response_model=list[VoiceProfileOut])
def list_voice_profiles(project_id: str, db: Session = Depends(get_db)):
    speakers = db.query(Speaker).filter(Speaker.project_id == project_id).all()
    profiles = []
    for sp in speakers:
        profiles.extend(sp.voice_profiles)
    return profiles


# ── Voice Design Studio endpoints ─────────────────────────────────────────────

class VoicePreviewRequest(BaseModel):
    preset_id: Optional[str] = None
    worker_type: str = "custom"        # base / custom / design
    text: str = "こんにちは、よろしくお願いします。"
    language: str = "japanese"
    speaker_name: Optional[str] = None   # custom worker
    instruct: Optional[str] = None       # custom worker
    voice_description: Optional[str] = None  # design worker
    reference_audio_path: Optional[str] = None  # base worker
    reference_text: Optional[str] = None         # base worker
    speed: float = 1.0


@router.post("/{project_id}/speakers/{speaker_id}/preview_voice")
def preview_voice(
    project_id: str,
    speaker_id: str,
    body: VoicePreviewRequest,
    db: Session = Depends(get_db),
):
    """Generate a short voice preview. Returns a URL to stream the audio."""
    _get_speaker_or_404(project_id, speaker_id, db)

    if not body.text.strip():
        raise HTTPException(status_code=400, detail="テキストを入力してください")

    preview_id = str(uuid.uuid4())[:8]
    out_dir = _temp_dir() / speaker_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"preview_{preview_id}.wav"

    resolved = _resolve_profile_data_with_preset(body.model_dump(), db)

    try:
        from app.orchestrator.services.tts_dispatcher import synthesize
        result = synthesize(
            text=resolved.get("text", body.text),
            worker_type=resolved.get("worker_type", "custom"),
            output_path=out_path,
            language=resolved.get("language") or "japanese",
            speaker=resolved.get("speaker_name"),
            instruct=resolved.get("instruct"),
            voice_description=resolved.get("voice_description"),
            reference_audio_path=resolved.get("reference_audio_path"),
            reference_text=resolved.get("reference_text"),
            speed=float(resolved.get("speed") or 1.0),
        )
    except Exception as exc:
        logger.error(f"Voice preview synthesis failed: {exc}")
        raise HTTPException(status_code=500, detail=f"音声合成に失敗しました: {exc}")

    if not result.success:
        raise HTTPException(
            status_code=500,
            detail=f"音声合成ワーカーがエラーを返しました: {result.error or '不明なエラー'}"
        )

    return {
        "preview_url": f"/voice_preview/{speaker_id}/preview_{preview_id}.wav",
        "duration_seconds": result.duration_seconds,
        "worker_type": resolved.get("worker_type", "custom"),
    }


@router.post("/{project_id}/speakers/{speaker_id}/upload_reference")
async def upload_reference_audio(
    project_id: str,
    speaker_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload a reference audio file for voice cloning."""
    _get_speaker_or_404(project_id, speaker_id, db)

    fname = file.filename or "reference.wav"
    ext = Path(fname).suffix.lower()
    if ext not in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}:
        raise HTTPException(status_code=400, detail=f"Unsupported audio format: {ext}")

    ref_dir = get_data_dir() / "projects" / project_id / "references"
    ref_dir.mkdir(parents=True, exist_ok=True)
    uid = str(uuid.uuid4())[:8]
    dest = ref_dir / f"{speaker_id}_{uid}{ext}"

    data = await file.read()
    dest.write_bytes(data)
    logger.info(f"Saved reference audio: {dest}")

    return {
        "reference_audio_path": str(dest),
        "filename": dest.name,
        "size_bytes": len(data),
    }


class VoiceSuggestRequest(BaseModel):
    sample_count: int = 8  # how many dialogue samples to analyse


@router.post("/{project_id}/speakers/{speaker_id}/suggest_voice")
def suggest_voice(
    project_id: str,
    speaker_id: str,
    body: VoiceSuggestRequest,
    db: Session = Depends(get_db),
):
    """Use LLM to suggest voice descriptions for a speaker based on their dialogue."""
    sp = _get_speaker_or_404(project_id, speaker_id, db)

    # Gather dialogue samples
    segs = (
        db.query(Segment)
        .filter(
            Segment.project_id == project_id,
            Segment.segment_type.in_(["dialogue", "thought"]),
        )
        .order_by(Segment.order_index)
        .limit(body.sample_count * 4)
        .all()
    )

    # Filter to segments assigned to this speaker
    speaker_name = sp.name
    speaker_segs = [
        s for s in segs
        if (s.final_speaker or s.predicted_speaker or "") == speaker_name
    ][:body.sample_count]

    if not speaker_segs:
        # Fall back to first N segments of any type mentioning the name
        speaker_segs = segs[:body.sample_count]

    dialogue_text = "\n".join(
        f"- {s.normalized_text[:120]}" for s in speaker_segs
    )
    if not dialogue_text:
        dialogue_text = "（セリフなし）"

    llm_url = os.environ.get("LLM_API_URL", "")
    if not llm_url:
        return {
            "suggestions": [],
            "llm_available": False,
            "reason": "LLMサーバーが起動していません。LLM管理画面でモデルをロードしてください。",
        }

    prompt = f"""以下は小説のキャラクター「{speaker_name}」のセリフです。
このキャラクターの声の特徴を3案、JSON配列で提案してください。
各要素は以下の形式にしてください:
{{"worker_type": "custom" または "design", "speaker_name": "Qwen3-TTSの話者名またはnull", "instruct": "音声指示またはnull", "voice_description": "声の説明文（design時）またはnull", "reason": "この提案の根拠"}}

Qwen3-TTSの使用可能な話者: Aria, Roger, Sarah, Laura, Charlie, George, Callum, River, Liam, Charlotte, Alice, Matilda, Will, Jessica, Eric, Chris, Brian

セリフ例:
{dialogue_text}

JSONのみ返してください。"""

    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("LLM_API_KEY", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        import requests as req_lib
        resp = req_lib.post(
            f"{llm_url.rstrip('/')}/chat/completions",
            json={
                "model": os.environ.get("LLM_MODEL", "gpt-4o-mini"),
                "messages": [
                    {"role": "system", "content": "あなたは音声デザインの専門家です。JSONのみ返してください。"},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.7,
                "max_tokens": 800,
            },
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        # Strip markdown code blocks if present
        import re
        raw = re.sub(r"```json\s*", "", raw)
        raw = re.sub(r"```\s*", "", raw)
        import json as _json
        suggestions = _json.loads(raw)
        if not isinstance(suggestions, list):
            suggestions = [suggestions]
        # Touch last_used_at in llm_manager
        try:
            from app.orchestrator.services.llm_manager import touch_last_used
            touch_last_used()
        except Exception:
            pass
        return {"suggestions": suggestions[:3], "llm_available": True}
    except Exception as exc:
        logger.warning(f"Voice suggestion LLM call failed: {exc}")
        return {
            "suggestions": [],
            "llm_available": True,
            "error": str(exc),
        }


# ── TTS worker proxy endpoints ─────────────────────────────────────────────────
# Registered separately via tts_router with /api/tts prefix.

tts_router = APIRouter(prefix="/api/tts", tags=["tts_proxy"])


@tts_router.get("/{worker_type}/health")
def tts_worker_health(worker_type: str):
    """Proxy health check to TTS worker."""
    from app.orchestrator.services.tts_dispatcher import get_worker_url
    import requests as req_lib
    url = get_worker_url(worker_type)
    try:
        r = req_lib.get(f"{url}/health", timeout=5)
        return r.json()
    except Exception as exc:
        return {"status": "unreachable", "error": str(exc)}


@tts_router.get("/custom/speakers")
def tts_custom_speakers():
    """Proxy the custom TTS worker speaker list through orchestrator."""
    from app.orchestrator.services.tts_dispatcher import get_worker_url
    import requests as req_lib
    url = get_worker_url("custom")
    try:
        r = req_lib.get(f"{url}/speakers", timeout=5)
        if r.status_code == 200:
            return r.json()
        return []
    except Exception:
        # Return default known speaker list as fallback
        return [
            "Aria", "Roger", "Sarah", "Laura", "Charlie", "George",
            "Callum", "River", "Liam", "Charlotte", "Alice", "Matilda",
            "Will", "Jessica", "Eric", "Chris", "Brian",
        ]
