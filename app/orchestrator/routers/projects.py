"""Project CRUD API."""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session

from app.shared.database import get_db
from app.shared.models import Project
from app.shared.schemas import ProjectCreate, ProjectOut
from app.shared.logger import get_logger
from app.shared.paths import get_data_dir
from app.orchestrator.services.ingest import UPLOAD_EXTENSIONS, ingest_uploaded_file

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
    project = _get_project_or_404(project_id, db)
    filename = file.filename or "upload.bin"
    ext = Path(filename).suffix.lower()
    if ext not in UPLOAD_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported extension: {ext}")

    project_dir = _projects_dir() / project_id
    temp_dir = project_dir / "temp_ingest"
    try:
        summary = ingest_uploaded_file(file, project_dir=project_dir, temp_root=temp_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Ingest failed")
        raise HTTPException(status_code=500, detail=f"Ingest failed: {exc}") from exc

    dest = project_dir / summary.normalized_filename
    raw_text = dest.read_text(encoding="utf-8")
    raw_text_preview = raw_text[:300]
    logger.info(
        f"Upload normalized text saved: project={project_id} path={dest} "
        f"upload_size_bytes={summary.upload_size_bytes} preview={raw_text_preview!r}"
    )
    project.raw_text_path = str(dest)
    project.uploaded_filename = summary.original_upload_name
    project.status = "uploaded"
    db.commit()
    db.refresh(project)
    logger.info(f"Uploaded source to project {project_id}: {dest}")
    return {
        "project_id": project_id,
        "path": str(dest),
        "char_count": summary.char_count,
        "upload_size_bytes": summary.upload_size_bytes,
        "source_type": summary.source_type,
        "raw_text_path": str(dest),
        "raw_text_preview": raw_text_preview,
        "extracted_files": summary.extracted_files,
        "warnings": summary.warnings,
    }
