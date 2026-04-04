"""Voice mapping API."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.shared.database import get_db
from app.shared.models import Speaker, VoiceProfile
from app.shared.schemas import SpeakerOut, VoiceMappingBatch, VoiceProfileOut
from app.shared.logger import get_logger

logger = get_logger("router.voice_mapping")
router = APIRouter(prefix="/api/projects", tags=["voice_mapping"])


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
        profile_data = entry.profile.model_dump()
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
