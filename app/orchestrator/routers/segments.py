"""Segment retrieval and update API."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.shared.database import get_db
from app.shared.models import Segment
from app.shared.schemas import SegmentOut, SegmentsBatchUpdate
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
    updated = 0
    for update in body.updates:
        seg_id = update.get("id")
        if not seg_id:
            continue
        seg = db.get(Segment, seg_id)
        if not seg or seg.project_id != project_id:
            continue
        if "final_speaker" in update:
            seg.final_speaker = update["final_speaker"]
        if "segment_type" in update:
            seg.segment_type = update["segment_type"]
        updated += 1
    db.commit()
    return {"updated": updated}
