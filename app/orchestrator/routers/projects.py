"""Project CRUD API."""
from __future__ import annotations

import io
import json
import threading
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session

from app.shared.database import get_db, SessionLocal
from app.shared.models import Project, Segment
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


@router.post("/{project_id}/upload_text")
def upload_text(
    project_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload and ingest a file. Returns upload_id for SSE progress tracking.

    The actual ingest runs in a background thread. Callers can:
    - Poll ``GET /api/projects/{id}/upload/progress?upload_id=...`` for SSE events
    - Or wait for completion synchronously (backwards-compat: still returns on finish)
    """
    project = _get_project_or_404(project_id, db)
    filename = file.filename or "upload.bin"
    ext = Path(filename).suffix.lower()
    if ext not in UPLOAD_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported extension: {ext}")

    # Read file bytes NOW (before background thread, to avoid UploadFile closing)
    raw_bytes: bytes = file.file.read()

    upload_id = str(uuid.uuid4())
    ip.create(upload_id)

    project_dir = _projects_dir() / project_id
    temp_dir = project_dir / "temp_ingest"

    # Result holder for blocking return
    _result: dict = {}
    _done_event = threading.Event()

    def _bg_ingest() -> None:
        import io as _io
        from fastapi import UploadFile as _UF

        # Reconstruct a minimal file-like object for ingest
        class _FakeSF:
            def __init__(self, data: bytes, fname: str):
                self.filename = fname
                self.file = _io.BytesIO(data)

        fake_file = _FakeSF(raw_bytes, filename)

        def _cb(stage: str, **kwargs) -> None:
            ip.update(upload_id, stage, **kwargs)

        try:
            summary = ingest_uploaded_file(fake_file, project_dir=project_dir, temp_root=temp_dir, progress_cb=_cb)
        except Exception as exc:
            logger.exception("Background ingest failed")
            ip.fail(upload_id, str(exc))
            _result["error"] = str(exc)
            _done_event.set()
            return

        # Update DB in a new session (background thread cannot reuse request session)
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
                _result["summary"] = summary
                _result["raw_text_preview"] = raw_text_preview
        except Exception as exc:
            logger.exception("DB update after ingest failed")
            ip.fail(upload_id, f"DB update failed: {exc}")
            _result["error"] = str(exc)
        finally:
            bg_db.close()

        ip.complete(upload_id)
        _done_event.set()

    thread = threading.Thread(target=_bg_ingest, daemon=True)
    thread.start()

    # Wait for background ingest to finish (synchronous path for backwards compat)
    _done_event.wait(timeout=300)

    if "error" in _result:
        raise HTTPException(status_code=500, detail=f"Ingest failed: {_result['error']}")

    summary = _result.get("summary")
    if not summary:
        raise HTTPException(status_code=500, detail="Ingest did not complete in time")

    dest = project_dir / summary.normalized_filename
    return {
        "project_id": project_id,
        "upload_id": upload_id,
        "path": str(dest),
        "char_count": summary.char_count,
        "upload_size_bytes": summary.upload_size_bytes,
        "source_type": summary.source_type,
        "raw_text_path": str(dest),
        "raw_text_preview": _result.get("raw_text_preview", ""),
        "extracted_files": summary.extracted_files,
        "warnings": summary.warnings,
    }


# ── Upload progress SSE endpoint ──────────────────────────────────────────────

@router.get("/{project_id}/upload/progress")
def upload_progress_sse(
    project_id: str,
    upload_id: str = Query(..., description="upload_id returned by upload_text"),
):
    """Server-Sent Events stream for upload/ingest progress.

    Send ``GET /api/projects/{id}/upload/progress?upload_id=...`` with an
    ``Accept: text/event-stream`` header to receive progress events.

    Each SSE event has ``data`` set to a JSON object:
      { stage, stage_label, page, total_pages, pct, warnings, status, error }

    The stream closes when status becomes "complete" or "failed".
    """
    def _generate():
        sent_complete = False
        for _ in range(600):  # max 600 × 0.5s = 5 minutes
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
            "X-Accel-Buffering": "no",  # disable nginx buffering
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
