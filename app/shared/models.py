from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    JSON,
    func,
)
from sqlalchemy.orm import relationship

from app.shared.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class Project(Base):
    __tablename__ = "projects"

    id = Column(String, primary_key=True, default=_uuid)
    name = Column(String, nullable=False)
    description = Column(Text, default="")
    status = Column(String, default="created")  # created/preprocessing/segmented/rendering/done/error
    raw_text_path = Column(String, nullable=True)
    uploaded_filename = Column(String, nullable=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    segments = relationship("Segment", back_populates="project", cascade="all, delete-orphan")
    speakers = relationship("Speaker", back_populates="project", cascade="all, delete-orphan")
    render_jobs = relationship("RenderJob", back_populates="project", cascade="all, delete-orphan")
    artifacts = relationship("Artifact", back_populates="project", cascade="all, delete-orphan")


class Segment(Base):
    __tablename__ = "segments"

    id = Column(String, primary_key=True, default=_uuid)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    chapter_index = Column(Integer, default=0)
    order_index = Column(Integer, nullable=False)
    raw_text = Column(Text, nullable=False)
    normalized_text = Column(Text, nullable=False)
    segment_type = Column(String, default="unknown")  # narration/dialogue/thought/unknown
    predicted_speaker = Column(String, default="unknown")
    confidence = Column(Float, default=0.0)
    reason = Column(Text, default="")
    final_speaker = Column(String, nullable=True)
    render_status = Column(String, default="pending")  # pending/running/done/error/skipped
    output_audio_path = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    project = relationship("Project", back_populates="segments")

    @property
    def effective_speaker(self) -> str:
        return self.final_speaker or self.predicted_speaker or "unknown"


class Speaker(Base):
    __tablename__ = "speakers"

    id = Column(String, primary_key=True, default=_uuid)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    name = Column(String, nullable=False)
    aliases = Column(JSON, default=list)  # list of alternative names
    segment_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=func.now())

    project = relationship("Project", back_populates="speakers")
    voice_profiles = relationship("VoiceProfile", back_populates="speaker", cascade="all, delete-orphan")


class VoiceProfile(Base):
    __tablename__ = "voice_profiles"

    id = Column(String, primary_key=True, default=_uuid)
    speaker_id = Column(String, ForeignKey("speakers.id"), nullable=False)
    worker_type = Column(String, default="custom")  # base/custom/design
    model_name = Column(String, nullable=True)
    preset_name = Column(String, nullable=True)
    language = Column(String, default="ja")
    speaker_name = Column(String, nullable=True)      # for custom worker
    instruct = Column(Text, nullable=True)            # for custom worker
    voice_description = Column(Text, nullable=True)  # for design worker
    reference_audio_path = Column(String, nullable=True)  # for base worker
    reference_text = Column(Text, nullable=True)     # for base worker
    speed = Column(Float, default=1.0)
    pause_ms_before = Column(Integer, default=0)
    pause_ms_after = Column(Integer, default=300)
    volume_gain_db = Column(Float, default=0.0)
    extra_params = Column(JSON, default=dict)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    speaker = relationship("Speaker", back_populates="voice_profiles")


class RenderJob(Base):
    __tablename__ = "render_jobs"

    id = Column(String, primary_key=True, default=_uuid)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    status = Column(String, default="pending")  # pending/running/done/error
    total_segments = Column(Integer, default=0)
    done_segments = Column(Integer, default=0)
    failed_segments = Column(Integer, default=0)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=func.now())

    project = relationship("Project", back_populates="render_jobs")


class Artifact(Base):
    __tablename__ = "artifacts"

    id = Column(String, primary_key=True, default=_uuid)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    artifact_type = Column(String, nullable=False)  # segment/chapter/full
    chapter_index = Column(Integer, nullable=True)
    file_path = Column(String, nullable=False)
    file_format = Column(String, default="wav")
    file_size_bytes = Column(Integer, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    created_at = Column(DateTime, default=func.now())

    project = relationship("Project", back_populates="artifacts")
