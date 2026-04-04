"""Segment retrieval and update API."""
from __future__ import annotations

import threading
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.shared.database import get_db, SessionLocal
from app.shared.models import Segment, ReviewLog
from app.shared.schemas import (
    SegmentOut, SegmentsBatchUpdate, ReviewLogOut, ConsistencyRepassRequest,
)
from app.shared.logger import get_logger

logger = get_logger("router.segments")
router = APIRouter(prefix="/api/projects", tags=["segments"])


@router.get("/{project_id}/segments", response_model=list[SegmentOut])
def list_segments(project_id: str, db: Session = Depends(get_db)):
    return (
        db.query(Segment)
        .filter(Segment.project_id == project_id)
        .order_by(Segment.chapter_index, Segment.order_index)
        .all()
    )


@router.post("/{project_id}/segments/update")
def update_segments(
    project_id: str,
    body: SegmentsBatchUpdate,
    db: Session = Depends(get_db),
):
    """Batch update final_speaker / segment_type.

    When final_speaker differs from predicted_speaker, a ReviewLog entry is
    automatically created for future fine-tuning / rule improvement.
    """
    updated = 0
    for update in body.updates:
        seg_id = update.get("id")
        if not seg_id:
            continue
        seg = db.get(Segment, seg_id)
        if not seg or seg.project_id != project_id:
            continue

        new_speaker = update.get("final_speaker")
        new_type = update.get("segment_type")
        correction_reason = update.get("correction_reason")

        # Write ReviewLog when predicted speaker is being overridden
        if (new_speaker is not None
                and new_speaker != seg.predicted_speaker
                and new_speaker.strip() != ""):
            log = ReviewLog(
                id=str(uuid.uuid4()),
                project_id=project_id,
                segment_id=seg_id,
                predicted_speaker=seg.predicted_speaker,
                predicted_confidence=seg.confidence,
                corrected_speaker=new_speaker,
                predicted_type=seg.segment_type,
                corrected_type=new_type or seg.segment_type,
                correction_reason=correction_reason,
                corrected_at=datetime.utcnow(),
            )
            db.add(log)

        if new_speaker is not None:
            seg.final_speaker = new_speaker
            # Once a human confirms a speaker, clear needs_review
            seg.needs_review = False
        if new_type is not None:
            seg.segment_type = new_type

        updated += 1

    db.commit()
    return {"updated": updated}


@router.get("/{project_id}/segments/{segment_id}/candidates")
def get_segment_candidates(
    project_id: str,
    segment_id: str,
    db: Session = Depends(get_db),
):
    """Return the candidate speakers for a single segment (for UI dropdown)."""
    seg = db.get(Segment, segment_id)
    if not seg or seg.project_id != project_id:
        raise HTTPException(status_code=404, detail="Segment not found")
    return seg.candidates or []


@router.post("/{project_id}/segments/consistency_repass", status_code=202)
def trigger_consistency_repass(
    project_id: str,
    body: ConsistencyRepassRequest,
    db: Session = Depends(get_db),
):
    """Re-run global consistency pass on a chapter (or all chapters) in background.

    Useful after a batch of human corrections to re-propagate changes to
    adjacent low-confidence segments.
    Returns a job_id that can be polled via GET /api/projects/{id}/jobs/{job_id}.
    """
    from app.shared.models import ProcessingJob, Project
    from app.orchestrator.routers.processing import _update_seg_job

    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    segs_q = db.query(Segment).filter(Segment.project_id == project_id)
    if body.chapter_index is not None:
        segs_q = segs_q.filter(Segment.chapter_index == body.chapter_index)
    total = segs_q.count()

    job = ProcessingJob(
        id=str(uuid.uuid4()),
        project_id=project_id,
        job_type="consistency_repass",
        status="running",
        stage="consistency_pass",
        stage_label="整合性チェック中",
        total_count=total,
        started_at=datetime.utcnow(),
    )
    db.add(job)
    db.commit()
    job_id = job.id

    chapter_filter = body.chapter_index
    threshold = body.threshold

    def _bg_repass():
        bg_db = SessionLocal()
        try:
            from app.orchestrator.services.speaker_segmenter import AnnotatedSegment
            from app.orchestrator.services.global_consistency import apply_global_consistency
            from app.orchestrator.services.character_builder import load_character_dict

            segs_q2 = (
                bg_db.query(Segment)
                .filter(Segment.project_id == project_id)
                .order_by(Segment.chapter_index, Segment.order_index)
            )
            if chapter_filter is not None:
                segs_q2 = segs_q2.filter(Segment.chapter_index == chapter_filter)
            db_segs = segs_q2.all()

            # Build AnnotatedSegment proxies (read-only metadata for consistency logic)
            proxy_list = []
            for s in db_segs:
                a = AnnotatedSegment(
                    chapter_index=s.chapter_index,
                    order_index=s.order_index,
                    raw_text=s.raw_text,
                    normalized_text=s.normalized_text,
                    segment_type=s.segment_type,
                    predicted_speaker=s.final_speaker or s.predicted_speaker,
                    confidence=s.confidence,
                    reason=s.reason or "",
                    needs_review=bool(s.needs_review) if s.needs_review is not None else False,
                )
                proxy_list.append(a)

            char_dict = load_character_dict(project_id, bg_db)
            apply_global_consistency(proxy_list, char_dict, threshold=threshold)

            # Write back updated confidence / needs_review
            proxy_map = {p.order_index: p for p in proxy_list}
            changed = 0
            for seg in db_segs:
                p = proxy_map.get(seg.order_index)
                if p and (p.confidence != seg.confidence or p.needs_review != seg.needs_review):
                    seg.confidence = p.confidence
                    seg.needs_review = p.needs_review
                    if p.predicted_speaker not in ("unknown", "") and not seg.final_speaker:
                        seg.predicted_speaker = p.predicted_speaker
                    changed += 1

            job_rec = bg_db.get(ProcessingJob, job_id)
            if job_rec:
                job_rec.status = "completed"
                job_rec.stage = "complete"
                job_rec.stage_label = "完了"
                job_rec.progress_pct = 100
                job_rec.finished_at = datetime.utcnow()
                job_rec.result_data = {"segments_updated": changed}

            bg_db.commit()
            logger.info(
                f"consistency_repass done: project={project_id} "
                f"chapter={chapter_filter} changed={changed}"
            )
        except Exception as exc:
            logger.exception(f"consistency_repass failed: project={project_id}")
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

    thread = threading.Thread(target=_bg_repass, daemon=True)
    thread.start()

    return {"job_id": job_id, "status": "running", "total_segments": total}


@router.get("/{project_id}/review_logs", response_model=list[ReviewLogOut])
def list_review_logs(
    project_id: str,
    limit: int = 200,
    db: Session = Depends(get_db),
):
    """List human corrections for this project (newest first)."""
    return (
        db.query(ReviewLog)
        .filter(ReviewLog.project_id == project_id)
        .order_by(ReviewLog.corrected_at.desc())
        .limit(limit)
        .all()
    )


@router.get("/{project_id}/export_training_data")
def export_training_data(project_id: str, db: Session = Depends(get_db)):
    """Export corrected segments as JSONL for fine-tuning / rule analysis.

    Returns a list of dicts (JSON), each representing one corrected segment
    with its context, candidates, and correction.
    """
    import json
    from fastapi.responses import StreamingResponse

    logs = (
        db.query(ReviewLog)
        .filter(ReviewLog.project_id == project_id)
        .order_by(ReviewLog.corrected_at)
        .all()
    )

    all_segs = (
        db.query(Segment)
        .filter(Segment.project_id == project_id)
        .order_by(Segment.order_index)
        .all()
    )
    seg_map = {s.id: s for s in all_segs}
    order_map = {s.order_index: s for s in all_segs}

    def _get_context(seg: Segment, n: int = 2) -> tuple[list, list]:
        before = []
        after = []
        for d in range(1, n + 1):
            s = order_map.get(seg.order_index - d)
            if s:
                before.insert(0, s.normalized_text[:100])
            s = order_map.get(seg.order_index + d)
            if s:
                after.append(s.normalized_text[:100])
        return before, after

    def _generate():
        for log in logs:
            seg = seg_map.get(log.segment_id)
            if not seg:
                continue
            ctx_before, ctx_after = _get_context(seg)
            record = {
                "segment_id": seg.id,
                "project_id": project_id,
                "chapter_index": seg.chapter_index,
                "order_index": seg.order_index,
                "raw_text": seg.raw_text,
                "normalized_text": seg.normalized_text,
                "segment_type": seg.segment_type,
                "context_before": ctx_before,
                "context_after": ctx_after,
                "candidates": seg.candidates or [],
                "evidence_spans": seg.evidence_spans or [],
                "predicted_speaker": log.predicted_speaker,
                "predicted_confidence": log.predicted_confidence,
                "corrected_speaker": log.corrected_speaker,
                "correction_reason": log.correction_reason,
                "corrected_at": log.corrected_at.isoformat() if log.corrected_at else None,
            }
            yield json.dumps(record, ensure_ascii=False) + "\n"

    return StreamingResponse(
        _generate(),
        media_type="application/x-ndjson",
        headers={"Content-Disposition": f"attachment; filename=training_{project_id}.jsonl"},
    )
