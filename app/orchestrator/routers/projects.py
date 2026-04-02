"""Project CRUD API."""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session

from app.shared.database import get_db
from app.shared.models import Project
from app.shared.schemas import ProjectCreate, ProjectOut
from app.shared.logger import get_logger
from app.shared.paths import get_data_dir
from app.orchestrator.services.ingest import load_text_file, save_project_text

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
    raw = file.file.read()

    # save to temp, detect encoding
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(raw)
        tmp_path = Path(tmp.name)

    try:
        text = load_text_file(tmp_path)
    finally:
        os.unlink(tmp_path)

    project_dir = _projects_dir() / project_id
    dest = save_project_text(project_dir, text, filename=file.filename or "upload.txt")

    project.raw_text_path = str(dest)
    project.status = "uploaded"
    db.commit()
    db.refresh(project)
    logger.info(f"Uploaded text to project {project_id}: {dest}")
    return {"project_id": project_id, "path": str(dest), "char_count": len(text)}
