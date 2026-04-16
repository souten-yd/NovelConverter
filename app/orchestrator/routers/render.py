"""Render and merge API."""
from __future__ import annotations

import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from app.shared.database import SessionLocal, get_db
from app.shared.models import Project, Segment, Speaker, VoiceProfile, VoicePreset, RenderJob, Artifact
from app.shared.schemas import RenderJobOut, ArtifactOut
from app.shared.logger import get_logger
from app.shared.paths import get_data_dir
from app.orchestrator.services.tts_dispatcher import synthesize
from app.orchestrator.services.audio_merger import merge_wavs, try_convert_to_m4b, get_wav_duration
from app.orchestrator.services.text_normalizer import prepare_tts_text

logger = get_logger("router.render")
router = APIRouter(prefix="/api/projects", tags=["render"])


def _output_dir() -> Path:
    return get_data_dir() / "outputs"


def _temp_dir() -> Path:
    return get_data_dir() / "temp"




def _resolve_render_profile(profile: Optional[VoiceProfile], db: Session) -> dict[str, object]:
    """Resolve synthesis parameters from preset_id at render time."""
    resolved = {
        "worker_type": "custom",
        "language": "japanese",
        "speaker_name": None,
        "instruct": None,
        "voice_description": None,
        "reference_audio_path": None,
        "reference_text": None,
        "speed": 1.0,
    }
    if not profile:
        return resolved

    resolved.update({
        "worker_type": profile.worker_type or "custom",
        "language": profile.language or "japanese",
        "speaker_name": profile.speaker_name,
        "instruct": profile.instruct,
        "voice_description": profile.voice_description,
        "reference_audio_path": profile.reference_audio_path,
        "reference_text": profile.reference_text,
        "speed": profile.speed if profile.speed is not None else 1.0,
    })

    if not profile.preset_id:
        return resolved

    preset = db.get(VoicePreset, profile.preset_id)
    if not preset:
        return resolved

    synth = preset.synthesis_params or {}
    resolved["worker_type"] = preset.engine_type or resolved["worker_type"]
    resolved["speaker_name"] = synth.get("speaker_name")
    resolved["instruct"] = synth.get("instruct")
    resolved["voice_description"] = synth.get("voice_description")
    resolved["reference_audio_path"] = synth.get("reference_audio_path")
    resolved["reference_text"] = synth.get("reference_text")
    if "language" in synth:
        resolved["language"] = synth.get("language")
    if "speed" in synth:
        resolved["speed"] = synth.get("speed")

    return resolved

def _get_project_or_404(project_id: str, db: Session) -> Project:
    p = db.get(Project, project_id)
    if not p:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    return p


def _render_background(project_id: str, job_id: str, skip_done: bool = True):
    """Background task: synthesize each segment."""
    db: Session = SessionLocal()
    try:
        job = db.get(RenderJob, job_id)
        if not job:
            return
        job.status = "running"
        job.started_at = datetime.utcnow()
        db.commit()

        segments = (
            db.query(Segment)
            .filter(Segment.project_id == project_id)
            .order_by(Segment.chapter_index, Segment.order_index)
            .all()
        )

        # Build speaker→profile map
        speakers = db.query(Speaker).filter(Speaker.project_id == project_id).all()
        sp_name_to_profile: dict[str, VoiceProfile] = {}
        for sp in speakers:
            if sp.voice_profiles:
                sp_name_to_profile[sp.name] = sp.voice_profiles[0]

        output_dir = _output_dir() / project_id / "segments"
        output_dir.mkdir(parents=True, exist_ok=True)

        job.total_segments = len(segments)
        db.commit()

        done = 0
        failed = 0

        for seg in segments:
            db.refresh(job)
            if job.status == "aborted":
                logger.info(f"Render job {job_id} aborted")
                break

            # Skip already-done if resume
            if skip_done and seg.render_status == "done" and seg.output_audio_path:
                if Path(seg.output_audio_path).exists():
                    done += 1
                    continue

            seg.render_status = "running"
            db.commit()

            speaker_name = seg.final_speaker or seg.predicted_speaker or "narrator"
            profile = sp_name_to_profile.get(speaker_name)
            tts_text = (seg.tts_text or "").strip() or prepare_tts_text(seg.ruby_text or seg.normalized_text)

            out_path = output_dir / f"seg_{seg.order_index:05d}.wav"

            resolved_profile = _resolve_render_profile(profile, db)

            try:
                result = synthesize(
                    text=tts_text,
                    worker_type=str(resolved_profile["worker_type"]),
                    output_path=out_path,
                    language=str(resolved_profile["language"]),
                    speaker=resolved_profile["speaker_name"],
                    instruct=resolved_profile["instruct"],
                    voice_description=resolved_profile["voice_description"],
                    reference_audio_path=resolved_profile["reference_audio_path"],
                    reference_text=resolved_profile["reference_text"],
                    speed=float(resolved_profile["speed"]),
                )
                if result.success and result.output_path:
                    seg.render_status = "done"
                    seg.output_audio_path = result.output_path
                    seg.tts_text = tts_text
                    seg.error_message = None
                    done += 1
                else:
                    seg.render_status = "error"
                    seg.error_message = result.error or "synthesis failed"
                    failed += 1
            except Exception as e:
                seg.render_status = "error"
                seg.error_message = str(e)
                failed += 1
                logger.error(f"Segment {seg.id} render error: {e}")

            job.done_segments = done
            job.failed_segments = failed
            db.commit()

        if job.status != "aborted":
            job.status = "done"
        job.finished_at = datetime.utcnow()
        db.commit()
        logger.info(f"Render job {job_id} ended: status={job.status} {done} ok, {failed} failed")
    except Exception as e:
        logger.error(f"Render job {job_id} crashed: {e}")
        job = db.get(RenderJob, job_id)
        if job:
            job.status = "error"
            job.error_message = str(e)
            db.commit()
    finally:
        db.close()


@router.post("/{project_id}/render", response_model=RenderJobOut, status_code=202)
def start_render(
    project_id: str,
    skip_done: bool = True,
    db: Session = Depends(get_db),
):
    project = _get_project_or_404(project_id, db)
    job = RenderJob(
        id=str(uuid.uuid4()),
        project_id=project_id,
        status="pending",
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    t = threading.Thread(
        target=_render_background,
        args=(project_id, job.id, skip_done),
        daemon=True,
    )
    t.start()
    return job


@router.get("/{project_id}/render/status", response_model=Optional[RenderJobOut])
def get_render_status(project_id: str, db: Session = Depends(get_db)):
    job = (
        db.query(RenderJob)
        .filter(RenderJob.project_id == project_id)
        .order_by(RenderJob.created_at.desc())
        .first()
    )
    return job


@router.post("/{project_id}/merge")
def merge_audio(project_id: str, db: Session = Depends(get_db)):
    project = _get_project_or_404(project_id, db)

    segments = (
        db.query(Segment)
        .filter(Segment.project_id == project_id, Segment.render_status == "done")
        .order_by(Segment.chapter_index, Segment.order_index)
        .all()
    )
    if not segments:
        raise HTTPException(status_code=400, detail="No rendered segments found")

    # Build speaker→profile map for pause settings
    speakers = db.query(Speaker).filter(Speaker.project_id == project_id).all()
    sp_profile_map: dict[str, VoiceProfile] = {}
    for sp in speakers:
        if sp.voice_profiles:
            sp_profile_map[sp.name] = sp.voice_profiles[0]

    out_dir = _output_dir() / project_id
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts_created = []

    # Per-chapter merge
    from itertools import groupby
    for chap_idx, chap_segs in groupby(segments, key=lambda s: s.chapter_index):
        chap_list = list(chap_segs)
        paths = [Path(s.output_audio_path) for s in chap_list if s.output_audio_path]
        if not paths:
            continue
        chap_out = out_dir / f"chapter_{chap_idx:03d}.wav"
        try:
            merge_wavs(paths, chap_out, pause_ms_before=0, pause_ms_after=300)
            duration = get_wav_duration(chap_out)
            art = Artifact(
                project_id=project_id,
                artifact_type="chapter",
                chapter_index=chap_idx,
                file_path=str(chap_out),
                file_format="wav",
                file_size_bytes=chap_out.stat().st_size,
                duration_seconds=duration,
            )
            db.add(art)
            artifacts_created.append(str(chap_out))
        except Exception as e:
            logger.error(f"Chapter {chap_idx} merge failed: {e}")

    # Full merge
    all_paths = [Path(s.output_audio_path) for s in segments if s.output_audio_path]
    full_out = out_dir / "full_audiobook.wav"
    try:
        merge_wavs(all_paths, full_out, pause_ms_before=0, pause_ms_after=500)
        duration = get_wav_duration(full_out)
        art = Artifact(
            project_id=project_id,
            artifact_type="full",
            chapter_index=None,
            file_path=str(full_out),
            file_format="wav",
            file_size_bytes=full_out.stat().st_size,
            duration_seconds=duration,
        )
        db.add(art)
        artifacts_created.append(str(full_out))

        # Try M4B
        m4b = try_convert_to_m4b(full_out)
        if m4b:
            art_m4b = Artifact(
                project_id=project_id,
                artifact_type="full",
                chapter_index=None,
                file_path=str(m4b),
                file_format="m4b",
                file_size_bytes=m4b.stat().st_size,
                duration_seconds=duration,
            )
            db.add(art_m4b)
            artifacts_created.append(str(m4b))
    except Exception as e:
        logger.error(f"Full merge failed: {e}")

    db.commit()
    return {"artifacts": artifacts_created}


@router.get("/{project_id}/artifacts", response_model=list[ArtifactOut])
def list_artifacts(project_id: str, db: Session = Depends(get_db)):
    return (
        db.query(Artifact)
        .filter(Artifact.project_id == project_id)
        .order_by(Artifact.created_at)
        .all()
    )


@router.post("/{project_id}/render/retry_failed")
def retry_failed(project_id: str, db: Session = Depends(get_db)):
    """Reset failed segments to pending so next render picks them up."""
    failed = (
        db.query(Segment)
        .filter(Segment.project_id == project_id, Segment.render_status == "error")
        .all()
    )
    for s in failed:
        s.render_status = "pending"
    db.commit()
    return {"reset": len(failed)}


@router.post("/{project_id}/render/speaker/{speaker_name}")
def render_speaker(
    project_id: str,
    speaker_name: str,
    db: Session = Depends(get_db),
):
    """Reset all segments for a given speaker to pending, then trigger render."""
    segs = (
        db.query(Segment)
        .filter(Segment.project_id == project_id)
        .all()
    )
    reset_count = 0
    for seg in segs:
        eff = seg.final_speaker or seg.predicted_speaker or "unknown"
        if eff == speaker_name:
            seg.render_status = "pending"
            reset_count += 1
    db.commit()

    # create new render job
    job = RenderJob(
        id=str(uuid.uuid4()),
        project_id=project_id,
        status="pending",
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    t = threading.Thread(
        target=_render_background,
        args=(project_id, job.id, True),
        daemon=True,
    )
    t.start()
    return {"reset": reset_count, "job_id": job.id}
