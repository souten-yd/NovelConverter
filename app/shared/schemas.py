from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Project ──────────────────────────────────────────────────────────────────

class ProjectCreate(BaseModel):
    name: str
    description: str = ""


class ProjectOut(BaseModel):
    id: str
    name: str
    description: str
    status: str
    raw_text_path: Optional[str]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ── Segment ───────────────────────────────────────────────────────────────────

class SegmentOut(BaseModel):
    id: str
    project_id: str
    chapter_index: int
    order_index: int
    raw_text: str
    normalized_text: str
    segment_type: str
    predicted_speaker: str
    confidence: float
    reason: str
    final_speaker: Optional[str]
    render_status: str
    output_audio_path: Optional[str]
    error_message: Optional[str]

    model_config = {"from_attributes": True}


class SegmentUpdate(BaseModel):
    final_speaker: Optional[str] = None
    segment_type: Optional[str] = None


class SegmentsBatchUpdate(BaseModel):
    updates: List[Dict[str, Any]]


# ── Speaker ───────────────────────────────────────────────────────────────────

class SpeakerOut(BaseModel):
    id: str
    project_id: str
    name: str
    aliases: List[str]
    segment_count: int

    model_config = {"from_attributes": True}


# ── VoiceProfile ──────────────────────────────────────────────────────────────

class VoiceProfileCreate(BaseModel):
    worker_type: str = "custom"
    model_name: Optional[str] = None
    preset_name: Optional[str] = None
    language: str = "ja"
    speaker_name: Optional[str] = None
    instruct: Optional[str] = None
    voice_description: Optional[str] = None
    reference_audio_path: Optional[str] = None
    reference_text: Optional[str] = None
    speed: float = 1.0
    pause_ms_before: int = 0
    pause_ms_after: int = 300
    volume_gain_db: float = 0.0
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class VoiceProfileOut(VoiceProfileCreate):
    id: str
    speaker_id: str

    model_config = {"from_attributes": True}


class VoiceMappingEntry(BaseModel):
    speaker_id: str
    profile: VoiceProfileCreate


class VoiceMappingBatch(BaseModel):
    mappings: List[VoiceMappingEntry]


# ── TTS Worker API ────────────────────────────────────────────────────────────

class SynthesizeRequest(BaseModel):
    text: str
    mode: str = "custom"  # custom / design / clone
    language: str = "ja"
    # custom
    speaker: Optional[str] = None
    instruct: Optional[str] = None
    # design
    voice_description: Optional[str] = None
    # clone
    reference_audio_path: Optional[str] = None
    reference_text: Optional[str] = None
    # common
    speed: float = 1.0
    output_path: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class SynthesizeResponse(BaseModel):
    success: bool
    output_path: Optional[str] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None
    worker_id: Optional[str] = None


class WorkerHealth(BaseModel):
    status: str
    worker_id: str
    worker_type: str
    model_loaded: bool
    version: str = "0.1.0"


# ── Render ────────────────────────────────────────────────────────────────────

class RenderJobOut(BaseModel):
    id: str
    project_id: str
    status: str
    total_segments: int
    done_segments: int
    failed_segments: int
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    error_message: Optional[str]

    model_config = {"from_attributes": True}


# ── Artifact ──────────────────────────────────────────────────────────────────

class ArtifactOut(BaseModel):
    id: str
    project_id: str
    artifact_type: str
    chapter_index: Optional[int]
    file_path: str
    file_format: str
    file_size_bytes: Optional[int]
    duration_seconds: Optional[float]
    created_at: datetime

    model_config = {"from_attributes": True}
