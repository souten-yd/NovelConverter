from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.orchestrator.services.tts_dispatcher import synthesize
from app.shared.database import get_db
from app.shared.models import VoicePreset
from app.shared.paths import get_data_dir
from app.shared.schemas import VoicePresetCreate, VoicePresetOut, VoicePresetUpdate

router = APIRouter(prefix="/api/voice-presets", tags=["voice_presets"])


def _sample_dir() -> Path:
    d = get_data_dir() / "voice_presets" / "samples"
    d.mkdir(parents=True, exist_ok=True)
    return d


@router.get("", response_model=list[VoicePresetOut])
def list_voice_presets(db: Session = Depends(get_db)):
    return db.query(VoicePreset).order_by(VoicePreset.updated_at.desc()).all()


@router.post("", response_model=VoicePresetOut)
def create_voice_preset(body: VoicePresetCreate, db: Session = Depends(get_db)):
    preset = VoicePreset(**body.model_dump())
    db.add(preset)
    db.commit()
    db.refresh(preset)
    return preset


@router.put("/{preset_id}", response_model=VoicePresetOut)
def update_voice_preset(preset_id: str, body: VoicePresetUpdate, db: Session = Depends(get_db)):
    preset = db.get(VoicePreset, preset_id)
    if not preset:
        raise HTTPException(status_code=404, detail="Preset not found")

    for k, v in body.model_dump(exclude_unset=True).items():
        setattr(preset, k, v)

    db.commit()
    db.refresh(preset)
    return preset


@router.delete("/{preset_id}")
def delete_voice_preset(preset_id: str, db: Session = Depends(get_db)):
    preset = db.get(VoicePreset, preset_id)
    if not preset:
        raise HTTPException(status_code=404, detail="Preset not found")
    db.delete(preset)
    db.commit()
    return {"deleted": 1, "preset_id": preset_id}


class VoicePresetPreviewRequest(BaseModel):
    text: str = "こんにちは、こちらはボイスプリセットの試聴です。"


@router.post("/{preset_id}/preview")
def preview_voice_preset(preset_id: str, body: VoicePresetPreviewRequest, db: Session = Depends(get_db)):
    preset = db.get(VoicePreset, preset_id)
    if not preset:
        raise HTTPException(status_code=404, detail="Preset not found")

    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="プレビューテキストを入力してください")

    sample_dir = _sample_dir()
    preview_id = str(uuid.uuid4())[:8]
    out_path = sample_dir / f"{preset_id}_{preview_id}.wav"

    synth = preset.synthesis_params or {}
    result = synthesize(
        text=text,
        worker_type=preset.engine_type,
        output_path=out_path,
        language=str(synth.get("language", "ja")),
        speaker=synth.get("speaker_name"),
        instruct=synth.get("instruct"),
        voice_description=synth.get("voice_description"),
        reference_audio_path=synth.get("reference_audio_path"),
        reference_text=synth.get("reference_text"),
        speed=float(synth.get("speed", 1.0) or 1.0),
    )

    if not result.success:
        raise HTTPException(status_code=500, detail=result.error or "音声合成に失敗しました")

    preset.sample_audio_path = str(out_path)
    db.commit()
    db.refresh(preset)

    return {
        "preset_id": preset.preset_id,
        "preview_url": f"/voice-presets/samples/{out_path.name}",
        "sample_audio_path": preset.sample_audio_path,
        "duration_seconds": result.duration_seconds,
    }
