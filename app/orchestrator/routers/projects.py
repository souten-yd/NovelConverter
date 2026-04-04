"""Project CRUD API."""
from __future__ import annotations

import io
import json
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session

from app.shared.database import get_db, SessionLocal
from app.shared.models import Project, ProcessingJob, Segment
from app.shared.schemas import ProjectCreate, ProjectOut
from app.shared.logger import get_logger
from app.shared.paths import get_data_dir
from app.orchestrator.services.ingest import UPLOAD_EXTENSIONS, ingest_uploaded_file
from app.orchestrator.services import ingest_progress as ip

logger = get_logger("router.projects")
router = APIRouter(prefix="/api/projects", tags=["projects"])


def _projects_dir() -> Path:
    return get_data_dir() / "projects"


def _get_project_or_404(project_id: str, db: Session) -> Project:
    p = db.get(Project, project_id)
    if not p:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    return p


@router.post("", response_model=ProjectOut, status_code=201)
def create_project(body: ProjectCreate, db: Session = Depends(get_db)):
    project = Project(id=str(uuid.uuid4()), name=body.name, description=body.description)
    db.add(project)
    db.commit()
    db.refresh(project)
    logger.info(f"Created project {project.id}: {project.name}")
    return project


@router.get("", response_model=list[ProjectOut])
def list_projects(db: Session = Depends(get_db)):
    return db.query(Project).order_by(Project.created_at.desc()).all()


@router.get("/{project_id}", response_model=ProjectOut)
def get_project(project_id: str, db: Session = Depends(get_db)):
    return _get_project_or_404(project_id, db)


@router.patch("/{project_id}")
def update_project(project_id: str, body: dict, db: Session = Depends(get_db)):
    """Update project settings (e.g. ocr_engine)."""
    project = _get_project_or_404(project_id, db)
    if "ocr_engine" in body:
        project.ocr_engine = body["ocr_engine"]
    if "name" in body:
        project.name = body["name"]
    if "description" in body:
        project.description = body["description"]
    db.commit()
    db.refresh(project)
    return {"id": project.id, "ocr_engine": project.ocr_engine}


@router.post("/{project_id}/upload_text", status_code=202)
def upload_text(
    project_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload and ingest a file asynchronously.

    Returns immediately with upload_id and job_id.
    Use SSE ``GET /api/projects/{id}/upload/progress?upload_id=...`` to track progress.
    Use ``GET /api/projects/{id}/jobs/{job_id}`` to query job status (reload-safe).
    """
    project = _get_project_or_404(project_id, db)
    filename = file.filename or "upload.bin"
    ext = Path(filename).suffix.lower()
    if ext not in UPLOAD_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"未対応の拡張子です: {ext}")

    # Read file bytes NOW (before background thread, to avoid UploadFile closing)
    raw_bytes: bytes = file.file.read()

    upload_id = str(uuid.uuid4())
    ip.create(upload_id)

    # Create persistent ProcessingJob record
    job = ProcessingJob(
        id=str(uuid.uuid4()),
        project_id=project_id,
        job_type="ingest",
        status="running",
        stage="upload",
        stage_label="アップロード受信中",
        started_at=datetime.utcnow(),
    )
    db.add(job)
    db.commit()
    job_id = job.id

    project_dir = _projects_dir() / project_id
    temp_dir = project_dir / "temp_ingest"
    ocr_engine = project.ocr_engine or "tesseract"

    def _bg_ingest() -> None:
        import io as _io

        class _FakeSF:
            def __init__(self, data: bytes, fname: str):
                self.filename = fname
                self.file = _io.BytesIO(data)

        fake_file = _FakeSF(raw_bytes, filename)

        def _cb(stage: str, **kwargs) -> None:
            ip.update(upload_id, stage, **kwargs)
            # Dual-write: persist to DB for reload recovery
            _update_job_from_progress(job_id, upload_id)

        try:
            summary = ingest_uploaded_file(
                fake_file, project_dir=project_dir, temp_root=temp_dir,
                progress_cb=_cb, ocr_engine=ocr_engine,
            )
        except Exception as exc:
            logger.exception("Background ingest failed")
            ip.fail(upload_id, str(exc))
            _fail_job(job_id, str(exc), _categorize_ingest_error(exc))
            return

        # Update DB in a new session
        bg_db = SessionLocal()
        try:
            proj = bg_db.get(Project, project_id)
            if proj:
                dest = project_dir / summary.normalized_filename
                raw_text_preview = ""
                if dest.exists():
                    raw_text = dest.read_text(encoding="utf-8")
                    raw_text_preview = raw_text[:300]
                    logger.info(
                        f"Upload normalized text saved: project={project_id} path={dest} "
                        f"upload_size_bytes={summary.upload_size_bytes} preview={raw_text_preview!r}"
                    )
                proj.raw_text_path = str(dest)
                proj.uploaded_filename = summary.original_upload_name
                proj.status = "uploaded"
                bg_db.commit()
                logger.info(f"Uploaded source to project {project_id}: {dest}")
        except Exception as exc:
            logger.exception("DB update after ingest failed")
            ip.fail(upload_id, f"DB update failed: {exc}")
            _fail_job(job_id, f"DB update failed: {exc}", "database")
            return
        finally:
            bg_db.close()

        ip.complete(upload_id)
        _complete_job(job_id, {
            "char_count": summary.char_count,
            "upload_size_bytes": summary.upload_size_bytes,
            "source_type": summary.source_type,
            "warnings": summary.warnings,
            "extracted_files_count": len(summary.extracted_files),
        })

    thread = threading.Thread(target=_bg_ingest, daemon=True)
    thread.start()

    return {
        "project_id": project_id,
        "upload_id": upload_id,
        "job_id": job_id,
        "status": "running",
    }


def _categorize_ingest_error(exc: Exception) -> str:
    """Classify an ingest exception into a user-facing error category."""
    msg = str(exc).lower()
    if "timeout" in msg or "timed out" in msg:
        return "timeout"
    if "rar" in msg or "unrar" in msg:
        return "extraction"
    if "zip" in msg or "archive" in msg:
        return "extraction"
    if "ocr" in msg or "tesseract" in msg or "pytesseract" in msg:
        return "ocr"
    if "encoding" in msg or "decode" in msg or "codec" in msg:
        return "encoding"
    if "permission" in msg or "disk" in msg or "space" in msg:
        return "filesystem"
    return "unknown"


def _update_job_from_progress(job_id: str, upload_id: str) -> None:
    """Sync in-memory progress to DB ProcessingJob."""
    data = ip.get(upload_id)
    if not data:
        return
    bg_db = SessionLocal()
    try:
        job = bg_db.get(ProcessingJob, job_id)
        if job and job.status == "running":
            job.stage = data.get("stage", "")
            job.stage_label = data.get("stage_label", "")
            job.progress_pct = data.get("pct", 0)
            job.current_count = data.get("page", 0)
            job.total_count = data.get("total_pages", 0)
            if data.get("warnings"):
                job.warnings = data["warnings"]
            bg_db.commit()
    except Exception:
        bg_db.rollback()
    finally:
        bg_db.close()


def _complete_job(job_id: str, result_data: dict) -> None:
    """Mark a ProcessingJob as completed."""
    bg_db = SessionLocal()
    try:
        job = bg_db.get(ProcessingJob, job_id)
        if job:
            job.status = "completed"
            job.stage = "complete"
            job.stage_label = "完了"
            job.progress_pct = 100
            job.result_data = result_data
            job.finished_at = datetime.utcnow()
            bg_db.commit()
    except Exception:
        bg_db.rollback()
    finally:
        bg_db.close()


def _fail_job(job_id: str, error_message: str, error_category: str = "unknown") -> None:
    """Mark a ProcessingJob as failed."""
    bg_db = SessionLocal()
    try:
        job = bg_db.get(ProcessingJob, job_id)
        if job:
            job.status = "failed"
            job.stage = "failed"
            job.stage_label = "失敗"
            job.error_message = error_message
            job.error_category = error_category
            job.finished_at = datetime.utcnow()
            bg_db.commit()
    except Exception:
        bg_db.rollback()
    finally:
        bg_db.close()


# ── Job status endpoints ─────────────────────────────────────────────────────

@router.get("/{project_id}/jobs")
def list_project_jobs(
    project_id: str,
    status: str = Query(None, description="Filter by status: running/completed/failed"),
    db: Session = Depends(get_db),
):
    """List processing jobs for a project. Used for reload recovery."""
    _get_project_or_404(project_id, db)
    q = db.query(ProcessingJob).filter(ProcessingJob.project_id == project_id)
    if status:
        q = q.filter(ProcessingJob.status == status)
    jobs = q.order_by(ProcessingJob.created_at.desc()).limit(20).all()
    return [
        {
            "id": j.id,
            "job_type": j.job_type,
            "status": j.status,
            "stage": j.stage,
            "stage_label": j.stage_label,
            "progress_pct": j.progress_pct,
            "current_count": j.current_count,
            "total_count": j.total_count,
            "warnings": j.warnings or [],
            "error_message": j.error_message,
            "error_category": j.error_category,
            "result_data": j.result_data or {},
            "started_at": j.started_at.isoformat() if j.started_at else None,
            "finished_at": j.finished_at.isoformat() if j.finished_at else None,
            "created_at": j.created_at.isoformat() if j.created_at else None,
        }
        for j in jobs
    ]


@router.get("/{project_id}/jobs/{job_id}")
def get_job_status(
    project_id: str,
    job_id: str,
    db: Session = Depends(get_db),
):
    """Get status of a specific processing job."""
    job = db.get(ProcessingJob, job_id)
    if not job or job.project_id != project_id:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません")
    return {
        "id": job.id,
        "job_type": job.job_type,
        "status": job.status,
        "stage": job.stage,
        "stage_label": job.stage_label,
        "progress_pct": job.progress_pct,
        "current_count": job.current_count,
        "total_count": job.total_count,
        "warnings": job.warnings or [],
        "error_message": job.error_message,
        "error_category": job.error_category,
        "result_data": job.result_data or {},
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
    }


# ── Upload progress SSE endpoint ──────────────────────────────────────────────

@router.get("/{project_id}/upload/progress")
def upload_progress_sse(
    project_id: str,
    upload_id: str = Query(..., description="upload_id returned by upload_text"),
):
    """Server-Sent Events stream for upload/ingest progress.

    Each SSE event has ``data`` set to a JSON object:
      { stage, stage_label, page, total_pages, pct, warnings, status, error }

    The stream closes when status becomes "complete" or "failed".
    """
    def _generate():
        sent_complete = False
        for _ in range(1200):  # max 1200 × 0.5s = 10 minutes
            data = ip.get(upload_id)
            if data is None:
                yield f"data: {json.dumps({'status':'not_found','stage':'','stage_label':'不明なupload_id','pct':0})}\n\n"
                return
            yield f"data: {json.dumps(data)}\n\n"
            if data["status"] in ("complete", "failed"):
                sent_complete = True
                break
            time.sleep(0.5)

        if not sent_complete:
            yield f"data: {json.dumps({'status':'timeout','stage':'failed','stage_label':'タイムアウト','pct':0,'error':'progress timeout'})}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ── Text download endpoints ───────────────────────────────────────────────────

@router.get("/{project_id}/download_text")
def download_project_text(
    project_id: str,
    variant: str = Query("normalized", description="normalized | manifest | segmented"),
    db: Session = Depends(get_db),
):
    """Download extracted / normalized text from a project.

    variant:
      - normalized  : the UTF-8 normalized text saved during upload
      - manifest    : ingest_manifest.json (file breakdown + warnings)
      - segmented   : segments joined in order (requires preprocessing)
    """
    project = _get_project_or_404(project_id, db)
    project_dir = _projects_dir() / project_id

    if variant == "normalized":
        if not project.raw_text_path:
            raise HTTPException(status_code=404, detail="テキストがまだアップロードされていません")
        text_path = Path(project.raw_text_path)
        if not text_path.exists():
            raise HTTPException(status_code=404, detail="正規化テキストファイルが見つかりません")
        safe_name = f"{project.name}_normalized.txt".replace("/", "_").replace("\\", "_")
        return FileResponse(
            str(text_path),
            media_type="text/plain; charset=utf-8",
            filename=safe_name,
            headers={"Content-Disposition": f'attachment; filename*=UTF-8\'\'{safe_name}'},
        )

    elif variant == "manifest":
        manifest_path = project_dir / "ingest_manifest.json"
        if not manifest_path.exists():
            raise HTTPException(status_code=404, detail="取り込みレポートが見つかりません")
        safe_name = f"{project.name}_manifest.json".replace("/", "_").replace("\\", "_")
        return FileResponse(
            str(manifest_path),
            media_type="application/json; charset=utf-8",
            filename=safe_name,
        )

    elif variant == "segmented":
        segs = (
            db.query(Segment)
            .filter(Segment.project_id == project_id)
            .order_by(Segment.chapter_index, Segment.order_index)
            .all()
        )
        if not segs:
            raise HTTPException(status_code=404, detail="セグメントがまだ生成されていません（前処理を実行してください）")

        lines: list[str] = []
        cur_chapter = -1
        for seg in segs:
            if seg.chapter_index != cur_chapter:
                cur_chapter = seg.chapter_index
                lines.append(f"\n===== 第{cur_chapter + 1}章 =====\n")
            speaker = seg.final_speaker or seg.predicted_speaker or "unknown"
            stype = {"narration": "地の文", "dialogue": "台詞", "thought": "心内", "unknown": "不明"}.get(seg.segment_type, seg.segment_type)
            lines.append(f"[{speaker} / {stype}] {seg.normalized_text}")

        combined = "\n".join(lines).strip() + "\n"
        safe_name = f"{project.name}_segments.txt".replace("/", "_").replace("\\", "_")
        return StreamingResponse(
            io.BytesIO(combined.encode("utf-8")),
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename*=UTF-8\'\'{safe_name}'},
        )

    else:
        raise HTTPException(status_code=400, detail=f"不明なvariant: {variant}。normalized / manifest / segmented のいずれかを指定してください")
