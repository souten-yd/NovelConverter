"""Preprocessing and speaker-segmentation endpoints."""
from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.shared.database import get_db, SessionLocal
from app.shared.models import Project, ProcessingJob, Segment, Speaker, VoiceProfile
from app.shared.schemas import SegmentOut, SpeakerOut
from app.shared.logger import get_logger
from app.orchestrator.services.preprocessor import preprocess, _normalize_whitespace, _split_chapters, _split_body
from app.orchestrator.services.speaker_segmenter import (
    segment_speakers,
    segment_speakers_enhanced,
    build_speaker_list,
    AVAILABLE_RULES,
    DiarizationConfig,
    RuleSpec,
    BATCH_SIZE,
)

logger = get_logger("router.processing")
router = APIRouter(prefix="/api/projects", tags=["processing"])


def _get_project_or_404(project_id: str, db: Session) -> Project:
    p = db.get(Project, project_id)
    if not p:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    return p


def _build_empty_segments_detail(project_dir: Path, raw_text: str) -> str:
    """Build an actionable error when preprocessing produced no segments."""
    detail = "テキストからセグメントが生成できませんでした。ファイルの内容を確認してください。"
    reasons: list[str] = []

    if not raw_text.strip():
        reasons.append("正規化後テキストが空です")

    manifest_path = project_dir / "ingest_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

        warnings = manifest.get("warnings") or []
        files = manifest.get("extracted_files") or []
        if any("RAR backend missing" in str(w) for w in warnings):
            reasons.append("RAR展開バックエンド(unrar/7zip/bsdtar)が見つかりません")
        if any("pytesseract not installed" in str(w) for w in warnings):
            reasons.append("pytesseract が未導入です")
        if any("OCR failed" in str(w) for w in warnings):
            reasons.append("OCR処理に失敗しました")
        if files and not any(f.get("status") == "ok" for f in files):
            reasons.append("入力ファイルは検出されましたが、有効な本文テキストを抽出できませんでした")

    if reasons:
        return f"{detail} 推定原因: {' / '.join(reasons)}"
    return detail


def _analyze_segmentation_stages(raw_text: str) -> tuple[list[str], dict[str, int]]:
    """Return likely causes and per-stage counts before DB persistence."""
    reasons: list[str] = []
    normalized = _normalize_whitespace(raw_text)
    chapters = _split_chapters(normalized) if normalized else []
    body_seg_count = 0
    for chap_idx, (_, body) in enumerate(chapters):
        body_seg_count += len(_split_body(body, chap_idx, 0))

    stats = {
        "raw_chars": len(raw_text),
        "raw_lines": raw_text.count("\n") + (1 if raw_text else 0),
        "trimmed_chars": len(raw_text.strip()),
        "normalized_chars": len(normalized),
        "chapter_count": len(chapters),
        "body_segment_candidates": body_seg_count,
    }

    if stats["trimmed_chars"] == 0:
        reasons.append("empty text")
    if raw_text and "\ufffd" in raw_text:
        reasons.append("decode failure")
    if stats["normalized_chars"] == 0 and stats["trimmed_chars"] > 0:
        reasons.append("parser removed all content")
    if stats["body_segment_candidates"] == 0 and stats["normalized_chars"] > 0:
        reasons.append("segmentation returned empty list")
    return reasons, stats


@router.post("/{project_id}/preprocess")
def preprocess_project(project_id: str, db: Session = Depends(get_db)):
    project = _get_project_or_404(project_id, db)
    if not project.raw_text_path:
        raise HTTPException(status_code=400, detail="No text uploaded yet")

    text_path = Path(project.raw_text_path)
    if not text_path.exists():
        raise HTTPException(status_code=400, detail="Text file missing on disk")

    raw_text = text_path.read_text(encoding="utf-8")
    preview = raw_text[:300]
    logger.info(f"Preprocess input: project={project_id} path={text_path} preview={preview!r}")
    causes, stage_stats = _analyze_segmentation_stages(raw_text)
    logger.info(
        "Preprocess stats: "
        f"project={project_id} chars={stage_stats['raw_chars']} lines={stage_stats['raw_lines']} "
        f"trimmed_chars={stage_stats['trimmed_chars']} normalized_chars={stage_stats['normalized_chars']} "
        f"chapters={stage_stats['chapter_count']} body_candidates={stage_stats['body_segment_candidates']}"
    )
    raw_segments, _norm_logs = preprocess(raw_text)

    if not raw_segments:
        project_dir = text_path.parent
        manifest_detail = _build_empty_segments_detail(project_dir, raw_text)
        if causes:
            manifest_detail = f"{manifest_detail} stage_causes={', '.join(causes)}"
        logger.warning(
            f"Preprocess produced 0 segments: project={project_id} "
            f"stage_causes={causes} stats={stage_stats}"
        )
        raise HTTPException(status_code=400, detail=manifest_detail)

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


class RuleSpecIn(BaseModel):
    rule_id: str
    enabled: bool = True
    params: dict = {}


class SegmentSpeakersRequest(BaseModel):
    rules: Optional[List[RuleSpecIn]] = None
    llm_fallback_mode: str = "on_error"  # disabled | on_error | always
    use_enhanced_pipeline: bool = False   # True → segment_speakers_enhanced()
    rebuild_char_dict: bool = True        # rebuild character dict before segmenting
    consistency_pass_threshold: float = 0.65


def _build_config(body: Optional[SegmentSpeakersRequest]) -> Optional[DiarizationConfig]:
    if body is None:
        return None
    if body.rules is None:
        return None
    return DiarizationConfig(
        rules=[RuleSpec(rule_id=r.rule_id, enabled=r.enabled, params=r.params) for r in body.rules],
        llm_fallback_mode=body.llm_fallback_mode,
    )


@router.post("/{project_id}/segment_speakers", status_code=202)
def segment_speakers_endpoint(
    project_id: str,
    body: Optional[SegmentSpeakersRequest] = None,
    db: Session = Depends(get_db),
):
    """Start speaker segmentation as an async job.

    Returns immediately with job_id. Poll via GET /api/projects/{id}/jobs/{job_id}.
    """
    project = _get_project_or_404(project_id, db)

    db_segs = (
        db.query(Segment)
        .filter(Segment.project_id == project_id)
        .order_by(Segment.order_index)
        .all()
    )
    if not db_segs:
        raise HTTPException(status_code=400, detail="前処理を先に実行してください")

    # Snapshot segment data for background thread
    seg_data = [
        {"chapter_index": s.chapter_index, "order_index": s.order_index, "text": s.normalized_text}
        for s in db_segs
    ]

    config = _build_config(body)
    total_segs = len(seg_data)

    # Create ProcessingJob
    job = ProcessingJob(
        id=str(uuid.uuid4()),
        project_id=project_id,
        job_type="speaker_segmentation",
        status="running",
        stage="rule_based",
        stage_label="ルールベース分類中",
        total_count=total_segs,
        started_at=datetime.utcnow(),
    )
    db.add(job)
    db.commit()
    job_id = job.id

    use_enhanced = bool(body and body.use_enhanced_pipeline)

    def _bg_segment():
        start_time = time.time()
        bg_db = SessionLocal()
        try:
            from app.orchestrator.services.preprocessor import RawSegment
            raw_segments = [
                RawSegment(chapter_index=s["chapter_index"], order_index=s["order_index"], text=s["text"])
                for s in seg_data
            ]

            _update_seg_job(bg_db, job_id, "rule_based", "ルールベース分類中", 10, 0, total_segs)
            logger.info(
                f"Speaker segmentation start: project={project_id} segments={total_segs} "
                f"enhanced={use_enhanced}"
            )

            def _pcb(stage, pct, cur=0, total=0):
                _update_seg_job(bg_db, job_id, stage, _seg_stage_label(stage), pct, cur, total)

            if use_enhanced:
                annotated, llm_errors = segment_speakers_enhanced(
                    raw_segments, config=config,
                    project_id=project_id, db=bg_db,
                    progress_cb=_pcb,
                )
            else:
                annotated, llm_errors = segment_speakers(
                    raw_segments, config=config, progress_cb=_pcb,
                )

            # Save results to DB
            _update_seg_job(bg_db, job_id, "saving", "結果保存中", 90, 0, total_segs)
            ann_map = {a.order_index: a for a in annotated}

            segs = (
                bg_db.query(Segment)
                .filter(Segment.project_id == project_id)
                .order_by(Segment.order_index)
                .all()
            )
            needs_review_count = 0
            for seg in segs:
                a = ann_map.get(seg.order_index)
                if a:
                    seg.segment_type = a.segment_type
                    seg.predicted_speaker = a.predicted_speaker
                    seg.confidence = a.confidence
                    seg.reason = a.reason
                    # Enhanced fields (no-op if columns don't exist on old schema,
                    # but _migrate_columns ensures they do)
                    if use_enhanced:
                        seg.candidates = a.candidates or []
                        seg.evidence_spans = a.evidence_spans or []
                        seg.needs_review = bool(a.needs_review)
                        seg.monologue_subtype = a.monologue_subtype
                        seg.rule_log = a.rule_log or []
                        if seg.needs_review:
                            needs_review_count += 1

            bg_db.query(Speaker).filter(Speaker.project_id == project_id).delete()
            speaker_list = build_speaker_list(annotated)
            for sp_data in speaker_list:
                sp = Speaker(
                    project_id=project_id,
                    name=sp_data["name"],
                    aliases=sp_data.get("aliases", []),
                    segment_count=sp_data["segment_count"],
                )
                bg_db.add(sp)
                bg_db.flush()
                vp = VoiceProfile(speaker_id=sp.id, worker_type="custom", language="ja")
                bg_db.add(vp)

            proj = bg_db.get(Project, project_id)
            if proj:
                proj.status = "segmented"

            # Complete job
            job_rec = bg_db.get(ProcessingJob, job_id)
            if job_rec:
                job_rec.status = "completed"
                job_rec.stage = "complete"
                job_rec.stage_label = "完了"
                job_rec.progress_pct = 100
                job_rec.finished_at = datetime.utcnow()
                job_rec.result_data = {
                    "segment_count": len(segs),
                    "speaker_count": len(speaker_list),
                    "llm_errors": llm_errors,
                    "llm_error_count": len(llm_errors),
                    "needs_review_count": needs_review_count,
                    "enhanced_pipeline": use_enhanced,
                    "elapsed_seconds": round(time.time() - start_time, 1),
                }

            bg_db.commit()
            elapsed = round(time.time() - start_time, 1)
            logger.info(
                f"Speaker segmentation done: project={project_id} speakers={len(speaker_list)} "
                f"needs_review={needs_review_count} llm_errors={len(llm_errors)} elapsed={elapsed}s"
            )
        except Exception as exc:
            logger.exception(f"Speaker segmentation failed: project={project_id}")
            bg_db.rollback()
            try:
                job_rec = bg_db.get(ProcessingJob, job_id)
                if job_rec:
                    job_rec.status = "failed"
                    job_rec.stage = "failed"
                    job_rec.stage_label = "失敗"
                    job_rec.error_message = str(exc)
                    job_rec.error_category = "llm" if "llm" in str(exc).lower() else "unknown"
                    job_rec.finished_at = datetime.utcnow()
                    bg_db.commit()
            except Exception:
                pass
        finally:
            bg_db.close()

    thread = threading.Thread(target=_bg_segment, daemon=True)
    thread.start()

    return {
        "project_id": project_id,
        "job_id": job_id,
        "status": "running",
        "segment_count": total_segs,
    }


def _seg_stage_label(stage: str) -> str:
    labels = {
        "rule_based": "ルールベース分類中",
        "llm_refinement": "LLM推定中",
        "llm_batch": "LLMバッチ処理中",
        "speaker_propagation": "話者伝搬中",
        "saving": "結果保存中",
        "complete": "完了",
        "failed": "失敗",
        # Enhanced pipeline stages
        "char_dict_build": "キャラクター辞書構築中",
        "candidate_gen": "候補話者生成中",
        "consistency_pass": "整合性チェック中",
    }
    return labels.get(stage, stage)


def _update_seg_job(
    bg_db: Session, job_id: str, stage: str, label: str, pct: int, cur: int, total: int
) -> None:
    try:
        job = bg_db.get(ProcessingJob, job_id)
        if job and job.status == "running":
            job.stage = stage
            job.stage_label = label
            job.progress_pct = pct
            job.current_count = cur
            job.total_count = total
            bg_db.commit()
    except Exception:
        bg_db.rollback()


# ── Diarization studio endpoints ──────────────────────────────────────────────

@router.get("/{project_id}/diarization_rules")
def get_diarization_rules(project_id: str):
    """Return the catalogue of available rules for the diarization studio."""
    return {"rules": AVAILABLE_RULES, "default_order": list(AVAILABLE_RULES.keys())}


class PreviewDiarizationRequest(BaseModel):
    rules: List[RuleSpecIn]
    llm_fallback_mode: str = "on_error"  # disabled | on_error | always
    use_enhanced_pipeline: bool = False   # True → segment_speakers_enhanced()
    consistency_pass_threshold: float = 0.65


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
        rules=[RuleSpec(rule_id=r.rule_id, enabled=r.enabled, params=r.params) for r in body.rules],
        llm_fallback_mode=body.llm_fallback_mode,
    )

    if body.use_enhanced_pipeline:
        from app.orchestrator.services.character_builder import load_character_dict
        char_dict = load_character_dict(project_id, db)
        annotated, llm_errors = segment_speakers_enhanced(
            raw_segments,
            config=config,
            char_dict=char_dict,
            project_id=project_id,
            db=db,
        )
    else:
        annotated, llm_errors = segment_speakers(raw_segments, config=config)

    segments_out = [
        {
            "chapter_index": a.chapter_index,
            "order_index": a.order_index,
            "raw_text": a.raw_text,
            "segment_type": a.segment_type,
            "predicted_speaker": a.predicted_speaker,
            "confidence": round(a.confidence, 3),
            "reason": a.reason,
            "is_chapter_header": a.is_chapter_header,
            "needs_review": getattr(a, "needs_review", False),
            "candidates": getattr(a, "candidates", []),
        }
        for a in annotated
    ]

    # Return as dict to include llm_errors alongside segments
    return {
        "segments": segments_out,
        "llm_errors": llm_errors,
        "llm_error_count": len(llm_errors),
    }
