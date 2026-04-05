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
    ocr_engine: str = "tesseract"
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
    ruby_text: Optional[str] = None
    ruby_metadata: List[Dict[str, Any]] = Field(default_factory=list)
    tts_text: Optional[str] = None
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
    correction_reason: Optional[str] = None  # stored in review_logs


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
    preset_id: Optional[str] = None
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


class VoicePresetCreate(BaseModel):
    name: str
    description: str = ""
    engine_type: str = "custom"
    synthesis_params: Dict[str, Any] = Field(default_factory=dict)
    reference_metadata: Dict[str, Any] = Field(default_factory=dict)
    sample_audio_path: Optional[str] = None


class VoicePresetUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    engine_type: Optional[str] = None
    synthesis_params: Optional[Dict[str, Any]] = None
    reference_metadata: Optional[Dict[str, Any]] = None
    sample_audio_path: Optional[str] = None


class VoicePresetOut(VoicePresetCreate):
    preset_id: str
    created_at: datetime
    updated_at: datetime

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


# ── Character master ──────────────────────────────────────────────────────────

class CharacterMasterOut(BaseModel):
    id: str
    project_id: str
    canonical_name: str
    aliases: List[str]
    first_person_pronouns: List[str]
    speech_style_hints: List[str]
    honorifics_used: List[str]
    gender_hint: Optional[str]
    role_hint: Optional[str]
    segment_count: int

    model_config = {"from_attributes": True}


class CharacterMasterCreate(BaseModel):
    canonical_name: str
    aliases: List[str] = []
    first_person_pronouns: List[str] = []
    speech_style_hints: List[str] = []
    honorifics_used: List[str] = []
    gender_hint: Optional[str] = None
    role_hint: Optional[str] = None


class CharacterMasterUpdate(BaseModel):
    canonical_name: Optional[str] = None
    aliases: Optional[List[str]] = None
    first_person_pronouns: Optional[List[str]] = None
    speech_style_hints: Optional[List[str]] = None
    honorifics_used: Optional[List[str]] = None
    gender_hint: Optional[str] = None
    role_hint: Optional[str] = None


# ── Review log ────────────────────────────────────────────────────────────────

class ReviewLogOut(BaseModel):
    id: str
    project_id: str
    segment_id: str
    predicted_speaker: Optional[str]
    predicted_confidence: Optional[float]
    corrected_speaker: Optional[str]
    predicted_type: Optional[str]
    corrected_type: Optional[str]
    correction_reason: Optional[str]
    corrected_at: datetime

    model_config = {"from_attributes": True}


# ── Enhanced pipeline request ─────────────────────────────────────────────────

class ConsistencyRepassRequest(BaseModel):
    chapter_index: Optional[int] = None   # None = all chapters
    threshold: float = 0.65
