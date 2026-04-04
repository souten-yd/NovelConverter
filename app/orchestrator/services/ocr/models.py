"""Data models for the enhanced OCR pipeline.

These dataclasses carry page-level metadata, OCR results, ruby annotations,
and debug information through every pipeline stage.  They are the single
source of truth for ordering (page_index) and ruby structure.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EngineType(str, Enum):
    NDLOCR_LITE = "ndlocr_lite"
    PADDLE_FAST = "paddle_fast"
    PADDLE_LAYOUT = "paddle_layout"
    FALLBACK = "fallback"


class RubyMode(str, Enum):
    NONE = "none"
    POSSIBLE = "possible"
    DETECTED = "detected"
    AUTO_RUBY = "auto_ruby"


class DisplayMode(str, Enum):
    TEXT_ONLY = "text_only"
    RUBY_PARALLEL = "ruby_parallel"
    DEBUG_OVERLAY = "debug_overlay"


# ---------------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------------

@dataclass
class PageJob:
    """A single page to be OCR-processed.  Created by the input normalizer."""
    source_doc_id: str
    source_name: str
    page_index: int
    image_path: str

    def to_dict(self) -> dict:
        return {
            "source_doc_id": self.source_doc_id,
            "source_name": self.source_name,
            "page_index": self.page_index,
            "image_path": self.image_path,
        }


# ---------------------------------------------------------------------------
# Bounding boxes & tokens
# ---------------------------------------------------------------------------

@dataclass
class BBox:
    """Axis-aligned bounding box [x_min, y_min, x_max, y_max]."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)

    def to_list(self) -> list[float]:
        return [self.x_min, self.y_min, self.x_max, self.y_max]

    @classmethod
    def from_list(cls, coords: list[float]) -> "BBox":
        return cls(coords[0], coords[1], coords[2], coords[3])

    @classmethod
    def from_polygon(cls, points: list[list[float]]) -> "BBox":
        """Create bbox from PaddleOCR polygon [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]."""
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return cls(min(xs), min(ys), max(xs), max(ys))


@dataclass
class OCRToken:
    """A single text token with bounding box and ordering info."""
    text: str
    bbox: BBox
    confidence: float = 0.0
    block_order: int = 0
    line_order: int = 0
    token_order: int = 0
    is_ruby_candidate: bool = False
    orientation: str = "horizontal"  # "horizontal" | "vertical"

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "bbox": self.bbox.to_list(),
            "confidence": round(self.confidence, 3),
            "block_order": self.block_order,
            "line_order": self.line_order,
            "token_order": self.token_order,
            "is_ruby_candidate": self.is_ruby_candidate,
            "orientation": self.orientation,
        }


# ---------------------------------------------------------------------------
# Ruby structures
# ---------------------------------------------------------------------------

@dataclass
class RubyAttachment:
    """A ruby annotation linked to its parent base text."""
    base_text: str
    ruby_text: str
    base_bbox: list[float]
    ruby_bbox: list[float]
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "base": self.base_text,
            "ruby": self.ruby_text,
            "base_bbox": self.base_bbox,
            "ruby_bbox": self.ruby_bbox,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class LineSegment:
    """A segment within a line (either base text or ruby)."""
    seg_type: str  # "base" | "ruby"
    text: str
    parent: Optional[str] = None  # for ruby: the base text it annotates

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"type": self.seg_type, "text": self.text}
        if self.parent:
            d["parent"] = self.parent
        return d


@dataclass
class StructuredLine:
    """A line of text with ruby structure and reading-order position."""
    line_id: str
    orientation: str = "horizontal"
    segments: list[LineSegment] = field(default_factory=list)
    block_order: int = 0
    line_order: int = 0

    def to_dict(self) -> dict:
        return {
            "line_id": self.line_id,
            "orientation": self.orientation,
            "segments": [s.to_dict() for s in self.segments],
            "block_order": self.block_order,
            "line_order": self.line_order,
        }


# ---------------------------------------------------------------------------
# Scheduling decision
# ---------------------------------------------------------------------------

@dataclass
class ScheduleDecision:
    """The scheduler's routing decision for a single page."""
    page_index: int
    engine: EngineType
    ruby_mode: RubyMode = RubyMode.NONE
    ruby_confidence: float = 0.0
    layout_complexity: float = 0.0

    def to_dict(self) -> dict:
        return {
            "page_index": self.page_index,
            "engine": self.engine.value,
            "ruby_mode": self.ruby_mode.value,
            "ruby_confidence": round(self.ruby_confidence, 3),
            "layout_complexity": round(self.layout_complexity, 3),
        }


# ---------------------------------------------------------------------------
# OCR page result (central output of the pipeline per page)
# ---------------------------------------------------------------------------

@dataclass
class OCRPageResult:
    """Complete OCR result for one page – the core data structure."""
    page_index: int
    engine: str = ""
    elapsed_ms: int = 0
    # Ruby detection
    ruby_detected: bool = False
    ruby_confidence: float = 0.0
    ruby_mode: str = "none"
    ruby_candidates_count: int = 0
    # Text outputs
    plain_text: str = ""
    ruby_text: str = ""       # e.g. "漢字(かんじ)"
    ruby_html: str = ""       # e.g. "<ruby>漢字<rt>かんじ</rt></ruby>"
    # Structured data
    tokens: list[OCRToken] = field(default_factory=list)
    ruby_attachments: list[RubyAttachment] = field(default_factory=list)
    lines: list[StructuredLine] = field(default_factory=list)
    # Error handling
    status: str = "ok"  # "ok" | "error" | "warning"
    error_message: str = ""
    warnings: list[str] = field(default_factory=list)
    # Metadata
    source_name: str = ""
    image_path: str = ""
    layout_complexity: float = 0.0
    # Debug
    debug: dict = field(default_factory=dict)

    # Timing helper
    _start_time: float = field(default=0.0, repr=False)

    def start_timer(self) -> None:
        self._start_time = time.monotonic()

    def stop_timer(self) -> None:
        if self._start_time > 0:
            self.elapsed_ms = int((time.monotonic() - self._start_time) * 1000)

    def to_dict(self) -> dict:
        return {
            "page_index": self.page_index,
            "engine": self.engine,
            "elapsed_ms": self.elapsed_ms,
            "ruby_detected": self.ruby_detected,
            "ruby_confidence": round(self.ruby_confidence, 3),
            "ruby_mode": self.ruby_mode,
            "ruby_candidates_count": self.ruby_candidates_count,
            "plain_text": self.plain_text,
            "ruby_text": self.ruby_text,
            "ruby_html": self.ruby_html,
            "tokens": [t.to_dict() for t in self.tokens],
            "ruby_attachments": [r.to_dict() for r in self.ruby_attachments],
            "lines": [ln.to_dict() for ln in self.lines],
            "status": self.status,
            "error_message": self.error_message,
            "warnings": self.warnings,
            "source_name": self.source_name,
            "layout_complexity": round(self.layout_complexity, 3),
            "debug": self.debug,
        }


# ---------------------------------------------------------------------------
# Full pipeline result
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """The complete output of the OCR pipeline for a job."""
    job_id: str
    pages: list[OCRPageResult] = field(default_factory=list)
    total_elapsed_ms: int = 0
    total_pages: int = 0
    engine_stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "pages": [p.to_dict() for p in self.pages],
            "total_elapsed_ms": self.total_elapsed_ms,
            "total_pages": self.total_pages,
            "engine_stats": self.engine_stats,
        }


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """User-configurable settings for the OCR pipeline."""
    # PaddleOCR settings
    paddle_device: str = "gpu:0"
    paddle_use_layout: bool = True
    paddle_max_workers: int = 2
    paddle_lang: str = "japan"
    # NDLOCR-Lite settings
    ndlocr_max_workers: int = 4
    # General
    default_engine: str = "ndlocr_lite"
    enable_ruby_detection: bool = True
    ruby_confidence_threshold: float = 0.5
    enable_cache: bool = True
    cache_dir: str = ""
    # Routing thresholds
    complexity_threshold: float = 0.5  # above this → paddle_layout
    ruby_route_threshold: float = 0.6  # above this ruby confidence → paddle_layout

    def to_dict(self) -> dict:
        return {
            "paddle_device": self.paddle_device,
            "paddle_use_layout": self.paddle_use_layout,
            "paddle_max_workers": self.paddle_max_workers,
            "paddle_lang": self.paddle_lang,
            "ndlocr_max_workers": self.ndlocr_max_workers,
            "default_engine": self.default_engine,
            "enable_ruby_detection": self.enable_ruby_detection,
            "ruby_confidence_threshold": self.ruby_confidence_threshold,
            "enable_cache": self.enable_cache,
            "complexity_threshold": self.complexity_threshold,
            "ruby_route_threshold": self.ruby_route_threshold,
        }
