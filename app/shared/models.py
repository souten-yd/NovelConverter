from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
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
    ocr_engine = Column(String, default="tesseract")  # tesseract/paddleocr/qwen_vl/ndlocr_lite
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    segments = relationship("Segment", back_populates="project", cascade="all, delete-orphan")
    speakers = relationship("Speaker", back_populates="project", cascade="all, delete-orphan")
    render_jobs = relationship("RenderJob", back_populates="project", cascade="all, delete-orphan")
    artifacts = relationship("Artifact", back_populates="project", cascade="all, delete-orphan")
    processing_jobs = relationship("ProcessingJob", back_populates="project", cascade="all, delete-orphan")
    character_master = relationship("CharacterMaster", back_populates="project", cascade="all, delete-orphan")
    normalization_logs = relationship("NormalizationLog", back_populates="project", cascade="all, delete-orphan")
    ocr_pages = relationship("OcrPage", back_populates="project", cascade="all, delete-orphan")
    ocr_page_versions = relationship("OcrPageVersion", back_populates="project", cascade="all, delete-orphan")
    review_logs = relationship("ReviewLog", back_populates="project", cascade="all, delete-orphan")


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
    # Enhanced pipeline fields (added by _migrate_columns at startup)
    candidates = Column(JSON, default=list)         # [{"name": str, "confidence": float, "evidence": str}]
    evidence_spans = Column(JSON, default=list)     # evidence text spans from LLM
    needs_review = Column(Boolean, default=False)   # flagged for human review
    monologue_subtype = Column(String, nullable=True)  # "inner" | None
    rule_log = Column(JSON, default=list)           # [{"rule": str, "fired": bool, "detail": str}]
    ruby_text = Column(Text, nullable=True)         # machine-readable ruby format e.g. [漢字|かんじ]
    ruby_metadata = Column(JSON, default=list)      # [{"surface","reading","reading_source","ruby_span"}]
    tts_text = Column(Text, nullable=True)          # reading-prioritized text for TTS
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
    preset_id = Column(String, ForeignKey("voice_presets.preset_id"), nullable=True)
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
    preset = relationship("VoicePreset", back_populates="voice_profiles")


class VoicePreset(Base):
    __tablename__ = "voice_presets"

    preset_id = Column(String, primary_key=True, default=_uuid)
    name = Column(String, nullable=False)
    description = Column(Text, default="")
    engine_type = Column(String, default="custom")  # base/custom/design
    synthesis_params = Column(JSON, default=dict)
    reference_metadata = Column(JSON, default=dict)
    sample_audio_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    voice_profiles = relationship("VoiceProfile", back_populates="preset")


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


class ProcessingJob(Base):
    """Generic job tracker for long-running operations (ingest, speaker segmentation, etc.)."""
    __tablename__ = "processing_jobs"

    id = Column(String, primary_key=True, default=_uuid)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    job_type = Column(String, nullable=False)  # ingest / speaker_segmentation
    status = Column(String, default="queued")  # queued/running/completed/failed
    stage = Column(String, default="")
    stage_label = Column(String, default="")
    progress_pct = Column(Integer, default=0)
    current_count = Column(Integer, default=0)
    total_count = Column(Integer, default=0)
    warnings = Column(JSON, default=list)
    error_message = Column(Text, nullable=True)
    error_category = Column(String, nullable=True)  # timeout/extraction/ocr/network/llm/unknown
    result_data = Column(JSON, default=dict)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    project = relationship("Project", back_populates="processing_jobs")


# ── Enhanced pipeline models ──────────────────────────────────────────────────

class CharacterMaster(Base):
    """Per-project character dictionary built from narration text."""
    __tablename__ = "character_master"

    id = Column(String, primary_key=True, default=_uuid)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    canonical_name = Column(String, nullable=False)
    aliases = Column(JSON, default=list)                 # list of str
    first_person_pronouns = Column(JSON, default=list)  # ["僕", "俺"] mapped to this char
    speech_style_hints = Column(JSON, default=list)     # ["〜だよ", "〜ます"] patterns
    honorifics_used = Column(JSON, default=list)        # chars this char addresses with honorifics
    gender_hint = Column(String, nullable=True)         # "male" | "female" | "unknown"
    role_hint = Column(String, nullable=True)           # "protagonist" | "heroine" | "support" | etc.
    segment_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    project = relationship("Project", back_populates="character_master")
    alias_entries = relationship("CharacterAlias", back_populates="character",
                                 cascade="all, delete-orphan")


class CharacterAlias(Base):
    """Maps alias strings → canonical CharacterMaster entry."""
    __tablename__ = "character_aliases"

    id = Column(String, primary_key=True, default=_uuid)
    character_id = Column(String, ForeignKey("character_master.id"), nullable=False)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    alias = Column(String, nullable=False, index=True)
    alias_type = Column(String, default="name")  # "name" | "pronoun" | "honorific"
    created_at = Column(DateTime, default=func.now())

    character = relationship("CharacterMaster", back_populates="alias_entries")


class NormalizationLog(Base):
    """Records before/after for each normalization rule that fired."""
    __tablename__ = "normalization_logs"

    id = Column(String, primary_key=True, default=_uuid)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    segment_order_index = Column(Integer, nullable=True)  # None = pre-segmentation (page-level)
    rule_name = Column(String, nullable=False)  # "ruby_removal" | "quotemark_unify" | etc.
    before_text = Column(Text, nullable=False)
    after_text = Column(Text, nullable=False)
    char_delta = Column(Integer, default=0)
    created_at = Column(DateTime, default=func.now())

    project = relationship("Project", back_populates="normalization_logs")


class OcrPage(Base):
    """Tracks per-page OCR confidence, ruby detection, and timing."""
    __tablename__ = "ocr_pages"

    id = Column(String, primary_key=True, default=_uuid)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    page_index = Column(Integer, nullable=False)
    source_filename = Column(String, nullable=True)
    ocr_engine = Column(String, nullable=False)
    confidence = Column(Float, nullable=True)       # engine-reported 0-1, or None
    char_count = Column(Integer, default=0)
    low_confidence = Column(Boolean, default=False) # flagged for human review
    warnings = Column(JSON, default=list)
    # Enhanced pipeline fields
    elapsed_ms = Column(Integer, default=0)           # OCR processing time
    ruby_detected = Column(Boolean, default=False)    # ruby annotations found
    ruby_confidence = Column(Float, default=0.0)      # ruby detection confidence 0-1
    ruby_mode = Column(String, default="none")        # none / possible / detected
    ruby_candidates_count = Column(Integer, default=0)
    plain_text = Column(Text, nullable=True)          # ruby-stripped text
    ruby_text = Column(Text, nullable=True)           # annotated: 漢字(かんじ)
    ruby_html = Column(Text, nullable=True)           # <ruby>漢字<rt>かんじ</rt></ruby>
    layout_complexity = Column(Float, default=0.0)    # 0-1 from scheduler
    ruby_attachments = Column(JSON, default=list)     # [{base, ruby, confidence}]
    structured_lines = Column(JSON, default=list)     # structured line data
    image_path = Column(String, nullable=True)        # path to page image
    status = Column(String, default="ok")             # ok / error / warning
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=func.now())

    project = relationship("Project", back_populates="ocr_pages")


class OcrPageVersion(Base):
    """Version snapshot of OCR page results before re-run/overwrite."""
    __tablename__ = "ocr_page_versions"

    id = Column(String, primary_key=True, default=_uuid)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    page_index = Column(Integer, nullable=False)
    version_group_id = Column(String, nullable=False, index=True)
    created_by_job_id = Column(String, nullable=True)
    snapshot = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, default=func.now())

    project = relationship("Project", back_populates="ocr_page_versions")


class ReviewLog(Base):
    """Stores human corrections for future fine-tuning / rule improvement."""
    __tablename__ = "review_logs"

    id = Column(String, primary_key=True, default=_uuid)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    segment_id = Column(String, ForeignKey("segments.id"), nullable=False)
    predicted_speaker = Column(String, nullable=True)
    predicted_confidence = Column(Float, nullable=True)
    corrected_speaker = Column(String, nullable=True)
    predicted_type = Column(String, nullable=True)
    corrected_type = Column(String, nullable=True)
    correction_reason = Column(Text, nullable=True)
    corrected_at = Column(DateTime, default=func.now())
    corrected_by = Column(String, default="user")

    project = relationship("Project", back_populates="review_logs")
    segment = relationship("Segment")
