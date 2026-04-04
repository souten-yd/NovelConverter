"""Character master CRUD API.

Provides endpoints to read, create, update, and delete character entries
built by the enhanced speaker pipeline.  Also exposes a rebuild endpoint
that triggers async character dictionary reconstruction from current segments.
"""
from __future__ import annotations

import threading
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.shared.database import get_db, SessionLocal
from app.shared.models import CharacterMaster, CharacterAlias, Segment, ProcessingJob, Project
from app.shared.schemas import (
    CharacterMasterOut,
    CharacterMasterCreate,
    CharacterMasterUpdate,
)
from app.shared.logger import get_logger

logger = get_logger("router.characters")
router = APIRouter(prefix="/api/projects", tags=["characters"])


def _get_project_or_404(project_id: str, db: Session) -> Project:
    p = db.get(Project, project_id)
    if not p:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    return p


@router.get("/{project_id}/character_master", response_model=List[CharacterMasterOut])
def list_characters(project_id: str, db: Session = Depends(get_db)):
    """Return all characters in the character master for a project."""
    _get_project_or_404(project_id, db)
    return (
        db.query(CharacterMaster)
        .filter(CharacterMaster.project_id == project_id)
        .order_by(CharacterMaster.segment_count.desc())
        .all()
    )


@router.get("/{project_id}/character_master/{char_id}", response_model=CharacterMasterOut)
def get_character(project_id: str, char_id: str, db: Session = Depends(get_db)):
    """Return a single character entry."""
    char = db.get(CharacterMaster, char_id)
    if not char or char.project_id != project_id:
        raise HTTPException(status_code=404, detail="Character not found")
    return char


@router.post("/{project_id}/character_master", status_code=201, response_model=CharacterMasterOut)
def create_character(
    project_id: str,
    body: CharacterMasterCreate,
    db: Session = Depends(get_db),
):
    """Manually create a new character entry."""
    _get_project_or_404(project_id, db)
    char = CharacterMaster(
        id=str(uuid.uuid4()),
        project_id=project_id,
        canonical_name=body.canonical_name,
        aliases=body.aliases,
        first_person_pronouns=body.first_person_pronouns,
        speech_style_hints=body.speech_style_hints,
        honorifics_used=body.honorifics_used,
        gender_hint=body.gender_hint,
        role_hint=body.role_hint,
        segment_count=0,
    )
    db.add(char)

    for alias in body.aliases:
        db.add(CharacterAlias(
            character_id=char.id, project_id=project_id,
            alias=alias, alias_type="name",
        ))
    for pronoun in body.first_person_pronouns:
        db.add(CharacterAlias(
            character_id=char.id, project_id=project_id,
            alias=pronoun, alias_type="pronoun",
        ))

    db.commit()
    db.refresh(char)
    return char


@router.patch("/{project_id}/character_master/{char_id}", response_model=CharacterMasterOut)
def update_character(
    project_id: str,
    char_id: str,
    body: CharacterMasterUpdate,
    db: Session = Depends(get_db),
):
    """Update character metadata (name, aliases, pronouns, speech style, etc.)."""
    char = db.get(CharacterMaster, char_id)
    if not char or char.project_id != project_id:
        raise HTTPException(status_code=404, detail="Character not found")

    if body.canonical_name is not None:
        char.canonical_name = body.canonical_name
    if body.aliases is not None:
        char.aliases = body.aliases
        # Rebuild alias entries
        db.query(CharacterAlias).filter(
            CharacterAlias.character_id == char_id,
            CharacterAlias.alias_type == "name",
        ).delete()
        for alias in body.aliases:
            db.add(CharacterAlias(
                character_id=char_id, project_id=project_id,
                alias=alias, alias_type="name",
            ))
    if body.first_person_pronouns is not None:
        char.first_person_pronouns = body.first_person_pronouns
        db.query(CharacterAlias).filter(
            CharacterAlias.character_id == char_id,
            CharacterAlias.alias_type == "pronoun",
        ).delete()
        for pronoun in body.first_person_pronouns:
            db.add(CharacterAlias(
                character_id=char_id, project_id=project_id,
                alias=pronoun, alias_type="pronoun",
            ))
    if body.speech_style_hints is not None:
        char.speech_style_hints = body.speech_style_hints
    if body.honorifics_used is not None:
        char.honorifics_used = body.honorifics_used
    if body.gender_hint is not None:
        char.gender_hint = body.gender_hint
    if body.role_hint is not None:
        char.role_hint = body.role_hint

    char.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(char)
    return char


@router.delete("/{project_id}/character_master/{char_id}", status_code=204)
def delete_character(project_id: str, char_id: str, db: Session = Depends(get_db)):
    """Delete a character entry and its aliases."""
    char = db.get(CharacterMaster, char_id)
    if not char or char.project_id != project_id:
        raise HTTPException(status_code=404, detail="Character not found")
    db.delete(char)
    db.commit()


@router.post("/{project_id}/character_master/rebuild", status_code=202)
def rebuild_character_dict(project_id: str, db: Session = Depends(get_db)):
    """Async: rebuild the character dictionary from current segments.

    Returns a job_id to poll for completion.
    """
    _get_project_or_404(project_id, db)

    total = db.query(Segment).filter(Segment.project_id == project_id).count()
    if total == 0:
        raise HTTPException(status_code=400, detail="No segments found. Run preprocess first.")

    job = ProcessingJob(
        id=str(uuid.uuid4()),
        project_id=project_id,
        job_type="char_dict_rebuild",
        status="running",
        stage="char_dict_build",
        stage_label="キャラクター辞書構築中",
        total_count=total,
        started_at=datetime.utcnow(),
    )
    db.add(job)
    db.commit()
    job_id = job.id

    def _bg():
        bg_db = SessionLocal()
        try:
            from app.orchestrator.services.speaker_segmenter import AnnotatedSegment
            from app.orchestrator.services.character_builder import (
                build_character_dict, persist_character_dict,
            )

            db_segs = (
                bg_db.query(Segment)
                .filter(Segment.project_id == project_id)
                .order_by(Segment.order_index)
                .all()
            )

            proxy_list = [
                AnnotatedSegment(
                    chapter_index=s.chapter_index,
                    order_index=s.order_index,
                    raw_text=s.raw_text,
                    normalized_text=s.normalized_text,
                    segment_type=s.segment_type,
                    predicted_speaker=s.final_speaker or s.predicted_speaker,
                    confidence=s.confidence,
                    reason=s.reason or "",
                )
                for s in db_segs
            ]

            existing_names = list({s.final_speaker or s.predicted_speaker for s in db_segs
                                    if (s.final_speaker or s.predicted_speaker)
                                    not in ("unknown", "narrator", "protagonist", "")})
            char_dict = build_character_dict(proxy_list, existing_names)
            persist_character_dict(char_dict, project_id, bg_db)

            job_rec = bg_db.get(ProcessingJob, job_id)
            if job_rec:
                job_rec.status = "completed"
                job_rec.stage = "complete"
                job_rec.stage_label = "完了"
                job_rec.progress_pct = 100
                job_rec.finished_at = datetime.utcnow()
                job_rec.result_data = {"character_count": len(char_dict.entries)}
            bg_db.commit()
            logger.info(
                f"char_dict_rebuild done: project={project_id} "
                f"chars={len(char_dict.entries)}"
            )
        except Exception as exc:
            logger.exception(f"char_dict_rebuild failed: project={project_id}")
            bg_db.rollback()
            try:
                job_rec = bg_db.get(ProcessingJob, job_id)
                if job_rec:
                    job_rec.status = "failed"
                    job_rec.error_message = str(exc)
                    job_rec.finished_at = datetime.utcnow()
                    bg_db.commit()
            except Exception:
                pass
        finally:
            bg_db.close()

    threading.Thread(target=_bg, daemon=True).start()
    return {"job_id": job_id, "status": "running"}
