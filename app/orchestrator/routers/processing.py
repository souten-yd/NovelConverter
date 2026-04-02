"""Preprocessing and speaker-segmentation endpoints."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.shared.database import get_db
from app.shared.models import Project, Segment, Speaker, VoiceProfile
from app.shared.schemas import SegmentOut, SpeakerOut
from app.shared.logger import get_logger
from app.orchestrator.services.preprocessor import preprocess
from app.orchestrator.services.speaker_segmenter import (
    segment_speakers,
    build_speaker_list,
    AVAILABLE_RULES,
    DiarizationConfig,
    RuleSpec,
)

logger = get_logger("router.processing")
router = APIRouter(prefix="/api/projects", tags=["processing"])


def _get_project_or_404(project_id: str, db: Session) -> Project:
    p = db.get(Project, project_id)
    if not p:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    return p


@router.post("/{project_id}/preprocess")
def preprocess_project(project_id: str, db: Session = Depends(get_db)):
    project = _get_project_or_404(project_id, db)
    if not project.raw_text_path:
        raise HTTPException(status_code=400, detail="No text uploaded yet")

    text_path = Path(project.raw_text_path)
    if not text_path.exists():
        raise HTTPException(status_code=400, detail="Text file missing on disk")

    raw_text = text_path.read_text(encoding="utf-8")
    raw_segments = preprocess(raw_text)

    # Clear old segments
    db.query(Segment).filter(Segment.project_id == project_id).delete()

    for rs in raw_segments:
        seg = Segment(
            project_id=project_id,
            chapter_index=rs.chapter_index,
            order_index=rs.order_index,
            raw_text=rs.text,
            normalized_text=rs.text,
            segment_type="unknown",
            predicted_speaker="unknown",
            confidence=0.0,
            reason="",
        )
        db.add(seg)

    project.status = "preprocessed"
    db.commit()
    logger.info(f"Preprocessed project {project_id}: {len(raw_segments)} segments")
    return {"project_id": project_id, "segment_count": len(raw_segments)}


class SegmentSpeakersRequest(BaseModel):
    rules: Optional[List[RuleSpecIn]] = None


@router.post("/{project_id}/segment_speakers")
def segment_speakers_endpoint(
    project_id: str,
    body: Optional[SegmentSpeakersRequest] = None,
    db: Session = Depends(get_db),
):
    project = _get_project_or_404(project_id, db)

    db_segs = (
        db.query(Segment)
        .filter(Segment.project_id == project_id)
        .order_by(Segment.order_index)
        .all()
    )
    if not db_segs:
        raise HTTPException(status_code=400, detail="Run preprocess first")

    from app.orchestrator.services.preprocessor import RawSegment
    raw_segments = [
        RawSegment(
            chapter_index=s.chapter_index,
            order_index=s.order_index,
            text=s.normalized_text,
        )
        for s in db_segs
    ]

    config: Optional[DiarizationConfig] = None
    if body and body.rules is not None:
        config = DiarizationConfig(
            rules=[RuleSpec(rule_id=r.rule_id, enabled=r.enabled, params=r.params) for r in body.rules]
        )

    annotated = segment_speakers(raw_segments, config=config)
    ann_map = {a.order_index: a for a in annotated}

    for seg in db_segs:
        a = ann_map.get(seg.order_index)
        if a:
            seg.segment_type = a.segment_type
            seg.predicted_speaker = a.predicted_speaker
            seg.confidence = a.confidence
            seg.reason = a.reason

    # Build/update speakers
    db.query(Speaker).filter(Speaker.project_id == project_id).delete()
    speaker_list = build_speaker_list(annotated)
    for sp_data in speaker_list:
        sp = Speaker(
            project_id=project_id,
            name=sp_data["name"],
            aliases=sp_data.get("aliases", []),
            segment_count=sp_data["segment_count"],
        )
        db.add(sp)
        db.flush()
        # Create default voice profile
        vp = VoiceProfile(speaker_id=sp.id, worker_type="custom", language="ja")
        db.add(vp)

    project.status = "segmented"
    db.commit()
    logger.info(f"Speaker segmentation done: {len(speaker_list)} speakers")
    return {"project_id": project_id, "segment_count": len(db_segs), "speaker_count": len(speaker_list)}


# ── Diarization studio endpoints ──────────────────────────────────────────────

@router.get("/{project_id}/diarization_rules")
def get_diarization_rules(project_id: str):
    """Return the catalogue of available rules for the diarization studio."""
    return {"rules": AVAILABLE_RULES, "default_order": list(AVAILABLE_RULES.keys())}


class RuleSpecIn(BaseModel):
    rule_id: str
    enabled: bool = True
    params: dict = {}


class PreviewDiarizationRequest(BaseModel):
    rules: List[RuleSpecIn]


@router.post("/{project_id}/preview_diarization")
def preview_diarization(
    project_id: str,
    body: PreviewDiarizationRequest,
    db: Session = Depends(get_db),
):
    """Run diarization with a custom rule config and return results WITHOUT saving to DB."""
    _get_project_or_404(project_id, db)

    db_segs = (
        db.query(Segment)
        .filter(Segment.project_id == project_id)
        .order_by(Segment.order_index)
        .all()
    )
    if not db_segs:
        raise HTTPException(status_code=400, detail="Run preprocess first")

    from app.orchestrator.services.preprocessor import RawSegment
    raw_segments = [
        RawSegment(
            chapter_index=s.chapter_index,
            order_index=s.order_index,
            text=s.normalized_text,
        )
        for s in db_segs
    ]

    config = DiarizationConfig(
        rules=[RuleSpec(rule_id=r.rule_id, enabled=r.enabled, params=r.params) for r in body.rules]
    )

    annotated = segment_speakers(raw_segments, config=config)

    return [
        {
            "chapter_index": a.chapter_index,
            "order_index": a.order_index,
            "raw_text": a.raw_text,
            "segment_type": a.segment_type,
            "predicted_speaker": a.predicted_speaker,
            "confidence": round(a.confidence, 3),
            "reason": a.reason,
            "is_chapter_header": a.is_chapter_header,
        }
        for a in annotated
    ]
