"""UI page routes (Jinja2 templates)."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.shared.database import get_db
from app.shared.models import Project, Segment, Speaker, RenderJob, Artifact, VoiceProfile
from app.shared.paths import get_data_dir

TEMPLATE_DIR = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

router = APIRouter(tags=["ui"])


@router.get("/llm", response_class=HTMLResponse)
def llm_manager_page(request: Request):
    return templates.TemplateResponse(request, "llm_manager.html", {})


@router.get("/", response_class=HTMLResponse)
def home(request: Request, db: Session = Depends(get_db)):
    projects = db.query(Project).order_by(Project.created_at.desc()).all()
    return templates.TemplateResponse(request, "projects_list.html", {"projects": projects})


@router.get("/projects/{project_id}", response_class=HTMLResponse)
def project_detail(project_id: str, request: Request, db: Session = Depends(get_db)):
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404)
    seg_count = db.query(Segment).filter(Segment.project_id == project_id).count()
    sp_count = db.query(Speaker).filter(Speaker.project_id == project_id).count()
    return templates.TemplateResponse(
        request, "project_detail.html",
        {"project": project, "seg_count": seg_count, "sp_count": sp_count},
    )


@router.get("/projects/{project_id}/diarization_studio", response_class=HTMLResponse)
def diarization_studio_page(project_id: str, request: Request, db: Session = Depends(get_db)):
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404)
    return templates.TemplateResponse(
        request, "diarization_studio.html",
        {"project": project},
    )


@router.get("/projects/{project_id}/segments", response_class=HTMLResponse)
def segments_page(project_id: str, request: Request, db: Session = Depends(get_db)):
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404)
    segments = (
        db.query(Segment)
        .filter(Segment.project_id == project_id)
        .order_by(Segment.chapter_index, Segment.order_index)
        .all()
    )
    return templates.TemplateResponse(
        request, "segments.html",
        {"project": project, "segments": segments},
    )


@router.get("/projects/{project_id}/voice_mapping", response_class=HTMLResponse)
def voice_mapping_page(project_id: str, request: Request, db: Session = Depends(get_db)):
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404)
    speakers = db.query(Speaker).filter(Speaker.project_id == project_id).all()
    speaker_profiles = []
    for sp in speakers:
        vp = sp.voice_profiles[0] if sp.voice_profiles else None
        speaker_profiles.append({"speaker": sp, "profile": vp})
    return templates.TemplateResponse(
        request, "voice_mapping.html",
        {"project": project, "speaker_profiles": speaker_profiles},
    )


@router.get("/projects/{project_id}/render", response_class=HTMLResponse)
def render_page(project_id: str, request: Request, db: Session = Depends(get_db)):
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404)
    job = (
        db.query(RenderJob)
        .filter(RenderJob.project_id == project_id)
        .order_by(RenderJob.created_at.desc())
        .first()
    )
    artifacts = db.query(Artifact).filter(Artifact.project_id == project_id).all()
    total = db.query(Segment).filter(Segment.project_id == project_id).count()
    done = db.query(Segment).filter(
        Segment.project_id == project_id, Segment.render_status == "done"
    ).count()
    failed = db.query(Segment).filter(
        Segment.project_id == project_id, Segment.render_status == "error"
    ).count()
    return templates.TemplateResponse(
        request, "render.html",
        {
            "project": project,
            "job": job,
            "artifacts": artifacts,
            "total_segments": total,
            "done_segments": done,
            "failed_segments": failed,
        },
    )


# ── Mobile UI routes ──────────────────────────────────────────────────────────

@router.get("/m/", response_class=HTMLResponse)
def mobile_home(request: Request, db: Session = Depends(get_db)):
    projects = db.query(Project).order_by(Project.created_at.desc()).all()
    return templates.TemplateResponse(request, "mobile_index.html", {"projects": projects})


@router.get("/m/projects/{project_id}", response_class=HTMLResponse)
def mobile_project_detail(project_id: str, request: Request, db: Session = Depends(get_db)):
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404)
    seg_count = db.query(Segment).filter(Segment.project_id == project_id).count()
    sp_count = db.query(Speaker).filter(Speaker.project_id == project_id).count()
    return templates.TemplateResponse(
        request, "mobile_project.html",
        {"project": project, "seg_count": seg_count, "sp_count": sp_count},
    )


@router.get("/m/llm", response_class=HTMLResponse)
def mobile_llm_page(request: Request):
    return templates.TemplateResponse(request, "mobile_llm.html", {})


@router.get("/artifacts/download/{project_id}/{filename}")
def download_artifact(project_id: str, filename: str, db: Session = Depends(get_db)):
    artifact_path = get_data_dir() / "outputs" / project_id / filename
    if not artifact_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(artifact_path), filename=filename)


@router.get("/segments/audio/{project_id}/{filename}")
def stream_segment_audio(project_id: str, filename: str):
    audio_path = get_data_dir() / "outputs" / project_id / "segments" / filename
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(str(audio_path), media_type="audio/wav")


@router.get("/voice_preview/{speaker_id}/{filename}")
def stream_voice_preview(speaker_id: str, filename: str):
    """Serve voice preview audio generated by the Voice Design Studio."""
    audio_path = get_data_dir() / "temp" / "voice_preview" / speaker_id / filename
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Preview audio not found")
    return FileResponse(str(audio_path), media_type="audio/wav")


@router.get("/projects/{project_id}/voice_studio", response_class=HTMLResponse)
def voice_studio_page(project_id: str, request: Request, db: Session = Depends(get_db)):
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404)
    speakers = db.query(Speaker).filter(Speaker.project_id == project_id).all()
    speaker_profiles = []
    for sp in speakers:
        vp = sp.voice_profiles[0] if sp.voice_profiles else None
        speaker_profiles.append({"speaker": sp, "profile": vp})
    return templates.TemplateResponse(
        request, "voice_studio.html",
        {"project": project, "speaker_profiles": speaker_profiles},
    )
