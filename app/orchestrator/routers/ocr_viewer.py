"""OCR Viewer API – endpoints for the parallel display UI.

Provides:
  - GET  /api/projects/{id}/ocr/pages          → list all OCR pages
  - GET  /api/projects/{id}/ocr/pages/{index}  → single page result
  - POST /api/projects/{id}/ocr/run_pipeline   → trigger enhanced pipeline
  - GET  /api/projects/{id}/ocr/pipeline_config → current config
  - POST /api/projects/{id}/ocr/pipeline_config → update config
"""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.shared.database import get_db, SessionLocal
from app.shared.logger import get_logger
from app.shared.models import OcrPage, ProcessingJob, Project

logger = get_logger("router.ocr_viewer")
router = APIRouter(prefix="/api/projects", tags=["ocr_viewer"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class PipelineConfigRequest(BaseModel):
    paddle_device: str = "gpu:0"
    paddle_use_layout: bool = True
    paddle_max_workers: int = 2
    paddle_lang: str = "japan"
    ndlocr_max_workers: int = 4
    default_engine: str = "ndlocr_lite"
    enable_ruby_detection: bool = True
    ruby_confidence_threshold: float = 0.5
    enable_cache: bool = True


class RunPipelineRequest(BaseModel):
    config: Optional[PipelineConfigRequest] = None
    force: bool = False  # re-run even if cached results exist


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_project_or_404(project_id: str, db: Session) -> Project:
    p = db.get(Project, project_id)
    if not p:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    return p


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/{project_id}/ocr/pages")
def list_ocr_pages(
    project_id: str,
    ruby_only: bool = Query(False, description="Only show pages with ruby detected"),
    db: Session = Depends(get_db),
):
    """List all OCR page results for a project."""
    _get_project_or_404(project_id, db)
    q = db.query(OcrPage).filter(OcrPage.project_id == project_id)
    if ruby_only:
        q = q.filter(OcrPage.ruby_detected == True)
    pages = q.order_by(OcrPage.page_index).all()

    return {
        "project_id": project_id,
        "total_pages": len(pages),
        "pages": [
            {
                "page_index": p.page_index,
                "source_filename": p.source_filename,
                "ocr_engine": p.ocr_engine,
                "elapsed_ms": p.elapsed_ms or 0,
                "char_count": p.char_count,
                "ruby_detected": p.ruby_detected,
                "ruby_confidence": p.ruby_confidence or 0,
                "ruby_mode": p.ruby_mode or "none",
                "ruby_candidates_count": p.ruby_candidates_count or 0,
                "layout_complexity": p.layout_complexity or 0,
                "status": p.status or "ok",
                "has_plain_text": bool(p.plain_text),
                "has_ruby_text": bool(p.ruby_text),
            }
            for p in pages
        ],
    }


@router.get("/{project_id}/ocr/pages/{page_index}")
def get_ocr_page(
    project_id: str,
    page_index: int,
    db: Session = Depends(get_db),
):
    """Get full OCR result for a single page, including all text formats."""
    _get_project_or_404(project_id, db)
    page = (
        db.query(OcrPage)
        .filter(OcrPage.project_id == project_id, OcrPage.page_index == page_index)
        .first()
    )
    if not page:
        raise HTTPException(status_code=404, detail=f"OCR page {page_index} not found")

    return {
        "page_index": page.page_index,
        "source_filename": page.source_filename,
        "ocr_engine": page.ocr_engine,
        "elapsed_ms": page.elapsed_ms or 0,
        "char_count": page.char_count,
        "confidence": page.confidence,
        "ruby_detected": page.ruby_detected,
        "ruby_confidence": page.ruby_confidence or 0,
        "ruby_mode": page.ruby_mode or "none",
        "ruby_candidates_count": page.ruby_candidates_count or 0,
        "layout_complexity": page.layout_complexity or 0,
        "plain_text": page.plain_text or "",
        "ruby_text": page.ruby_text or "",
        "ruby_html": page.ruby_html or "",
        "ruby_attachments": page.ruby_attachments or [],
        "structured_lines": page.structured_lines or [],
        "image_path": page.image_path,
        "status": page.status or "ok",
        "error_message": page.error_message,
        "warnings": page.warnings or [],
    }


@router.post("/{project_id}/ocr/run_pipeline", status_code=202)
def run_ocr_pipeline(
    project_id: str,
    body: Optional[RunPipelineRequest] = None,
    db: Session = Depends(get_db),
):
    """Trigger the enhanced OCR pipeline asynchronously.

    Returns job_id for progress tracking.
    """
    project = _get_project_or_404(project_id, db)

    if not project.raw_text_path:
        raise HTTPException(status_code=400, detail="テキストがまだアップロードされていません")

    # Create a processing job
    job = ProcessingJob(
        id=str(uuid.uuid4()),
        project_id=project_id,
        job_type="ocr_pipeline",
        status="running",
        stage="initializing",
        stage_label="OCRパイプライン初期化中",
        started_at=datetime.utcnow(),
    )
    db.add(job)
    db.commit()
    job_id = job.id

    # Build pipeline config
    from app.orchestrator.services.ocr.models import PipelineConfig
    config = PipelineConfig()
    if body and body.config:
        config.paddle_device = body.config.paddle_device
        config.paddle_use_layout = body.config.paddle_use_layout
        config.paddle_max_workers = body.config.paddle_max_workers
        config.paddle_lang = body.config.paddle_lang
        config.ndlocr_max_workers = body.config.ndlocr_max_workers
        config.default_engine = body.config.default_engine
        config.enable_ruby_detection = body.config.enable_ruby_detection
        config.ruby_confidence_threshold = body.config.ruby_confidence_threshold
        config.enable_cache = body.config.enable_cache

    force = body.force if body else False

    # Find image files from the project's ingest data
    project_dir = Path(project.raw_text_path).parent

    def _bg_run():
        bg_db = SessionLocal()
        try:
            _run_pipeline_bg(bg_db, project_id, job_id, project_dir, config, force)
        except Exception as exc:
            logger.exception(f"OCR pipeline failed: project={project_id}")
            try:
                job_rec = bg_db.get(ProcessingJob, job_id)
                if job_rec:
                    job_rec.status = "failed"
                    job_rec.stage = "failed"
                    job_rec.stage_label = "失敗"
                    job_rec.error_message = str(exc)
                    job_rec.error_category = "ocr"
                    job_rec.finished_at = datetime.utcnow()
                    bg_db.commit()
            except Exception:
                pass
        finally:
            bg_db.close()

    thread = threading.Thread(target=_bg_run, daemon=True)
    thread.start()

    return {
        "project_id": project_id,
        "job_id": job_id,
        "status": "running",
    }


def _run_pipeline_bg(
    db: Session,
    project_id: str,
    job_id: str,
    project_dir: Path,
    config,
    force: bool,
):
    """Background task: run the OCR pipeline and save results to DB."""
    import time
    from app.orchestrator.services.ocr.pipeline import run_pipeline
    from app.orchestrator.services.ingest import IMAGE_EXTENSIONS

    start = time.time()

    # Collect image files from project
    image_paths: list[Path] = []
    temp_dir = project_dir / "temp_ingest"
    extracted_dir = temp_dir / "extracted"

    # Check multiple locations for images
    for search_dir in [extracted_dir, temp_dir, project_dir]:
        if search_dir.exists():
            for p in search_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                    image_paths.append(p)
            if image_paths:
                break

    if not image_paths:
        _update_job(db, job_id, "failed", "失敗", "画像ファイルが見つかりません")
        return

    total_pages = len(image_paths)
    _update_job_progress(db, job_id, "scanning", "画像スキャン中", 5, 0, total_pages)

    # Progress callback
    def _cb(stage: str, cur: int, total: int, pct: int):
        stage_labels = {
            "normalizing": "入力正���化中",
            "scheduling": "エンジン割当中",
            "ocr": "OCR実行中",
            "postprocessing": "後処理中",
            "complete": "完了",
        }
        label = stage_labels.get(stage, stage)
        _update_job_progress(db, job_id, stage, label, pct, cur, total)

    # Run pipeline
    result = run_pipeline(
        image_paths=image_paths,
        config=config,
        job_id=f"ocr-{project_id[:8]}",
        progress_cb=_cb,
    )

    # Save results to DB
    _update_job_progress(db, job_id, "saving", "結果保存中", 95, 0, total_pages)

    # Clear old OCR pages
    db.query(OcrPage).filter(OcrPage.project_id == project_id).delete()

    for page in result.pages:
        ocr_page = OcrPage(
            project_id=project_id,
            page_index=page.page_index,
            source_filename=page.source_name,
            ocr_engine=page.engine,
            elapsed_ms=page.elapsed_ms,
            char_count=len(page.plain_text),
            ruby_detected=page.ruby_detected,
            ruby_confidence=page.ruby_confidence,
            ruby_mode=page.ruby_mode,
            ruby_candidates_count=page.ruby_candidates_count,
            plain_text=page.plain_text,
            ruby_text=page.ruby_text,
            ruby_html=page.ruby_html,
            layout_complexity=page.layout_complexity,
            ruby_attachments=[a.to_dict() for a in page.ruby_attachments],
            structured_lines=[ln.to_dict() for ln in page.lines],
            image_path=page.image_path,
            status=page.status,
            error_message=page.error_message if page.status == "error" else None,
            warnings=page.warnings,
        )
        db.add(ocr_page)

    # Complete job
    elapsed = round(time.time() - start, 1)
    job_rec = db.get(ProcessingJob, job_id)
    if job_rec:
        job_rec.status = "completed"
        job_rec.stage = "complete"
        job_rec.stage_label = "完了"
        job_rec.progress_pct = 100
        job_rec.finished_at = datetime.utcnow()
        job_rec.result_data = {
            "total_pages": result.total_pages,
            "total_elapsed_ms": result.total_elapsed_ms,
            "engine_stats": result.engine_stats,
            "elapsed_seconds": elapsed,
            "ruby_pages": sum(1 for p in result.pages if p.ruby_detected),
        }

    db.commit()
    logger.info(
        f"OCR pipeline saved: project={project_id} pages={result.total_pages} "
        f"elapsed={elapsed}s"
    )


def _update_job(db: Session, job_id: str, status: str, label: str, error: str = ""):
    try:
        job = db.get(ProcessingJob, job_id)
        if job:
            job.status = status
            job.stage = status
            job.stage_label = label
            if error:
                job.error_message = error
            job.finished_at = datetime.utcnow()
            db.commit()
    except Exception:
        db.rollback()


def _update_job_progress(
    db: Session, job_id: str, stage: str, label: str,
    pct: int, cur: int, total: int,
):
    try:
        job = db.get(ProcessingJob, job_id)
        if job and job.status == "running":
            job.stage = stage
            job.stage_label = label
            job.progress_pct = pct
            job.current_count = cur
            job.total_count = total
            db.commit()
    except Exception:
        db.rollback()
