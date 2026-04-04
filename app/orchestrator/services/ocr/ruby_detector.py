"""Ruby detection and attachment engine.

Detects ruby (furigana) annotations in OCR results by analysing the spatial
relationship between text bounding boxes.  Produces:

- A per-page ``RubyMode`` classification (none / possible / detected)
- ``RubyAttachment`` objects linking ruby text to base text
- Structured output in plain, annotated-text, and HTML formats

The detection is heuristic-based and provides confidence scores rather than
binary decisions, as specified in the requirements.
"""
from __future__ import annotations

import statistics
from typing import Optional

from app.shared.logger import get_logger
from app.orchestrator.services.ocr.models import (
    BBox,
    LineSegment,
    OCRPageResult,
    OCRToken,
    RubyAttachment,
    RubyMode,
    StructuredLine,
)

logger = get_logger("ocr.ruby_detector")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Height ratio thresholds: ruby text height is typically 20–55% of base text
RUBY_HEIGHT_RATIO_MIN = 0.15
RUBY_HEIGHT_RATIO_MAX = 0.60

# Max distance (pixels) between ruby box and base box edges
RUBY_PROXIMITY_MAX_PX = 40

# Min confidence to classify as "detected"
RUBY_DETECTED_THRESHOLD = 0.65
# Min confidence to classify as "possible"
RUBY_POSSIBLE_THRESHOLD = 0.35


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _classify_orientation(tokens: list[OCRToken]) -> str:
    """Guess dominant text orientation from token bounding boxes."""
    if not tokens:
        return "horizontal"

    vertical_score = 0
    horizontal_score = 0
    for t in tokens:
        if t.bbox.height > t.bbox.width * 1.3:
            vertical_score += 1
        else:
            horizontal_score += 1
    return "vertical" if vertical_score > horizontal_score else "horizontal"


def _compute_height_stats(tokens: list[OCRToken]) -> tuple[float, float]:
    """Return (median_height, std_height) of token bounding boxes."""
    heights = [t.bbox.height for t in tokens if t.bbox.height > 0]
    if not heights:
        return 0.0, 0.0
    med = statistics.median(heights)
    std = statistics.stdev(heights) if len(heights) > 1 else 0.0
    return med, std


def _is_small_box(token: OCRToken, median_h: float) -> bool:
    """Check if a token's height is in the ruby-size range relative to median."""
    if median_h <= 0:
        return False
    ratio = token.bbox.height / median_h
    return RUBY_HEIGHT_RATIO_MIN <= ratio <= RUBY_HEIGHT_RATIO_MAX


def _proximity_score(ruby_bbox: BBox, base_bbox: BBox, orientation: str) -> float:
    """Score how likely the ruby box is attached to the base box (0–1).

    For vertical text: ruby is to the right or above the base.
    For horizontal text: ruby is above the base.
    """
    # Check overlap along the reading axis
    if orientation == "vertical":
        # Ruby should be to the right of the base text
        h_dist = ruby_bbox.x_min - base_bbox.x_max
        if h_dist < -RUBY_PROXIMITY_MAX_PX or h_dist > RUBY_PROXIMITY_MAX_PX:
            return 0.0
        # Vertical overlap
        overlap_top = max(ruby_bbox.y_min, base_bbox.y_min)
        overlap_bot = min(ruby_bbox.y_max, base_bbox.y_max)
        overlap = max(0, overlap_bot - overlap_top)
        extent = min(ruby_bbox.height, base_bbox.height)
        if extent <= 0:
            return 0.0
        overlap_ratio = overlap / extent
        dist_factor = max(0, 1.0 - abs(h_dist) / RUBY_PROXIMITY_MAX_PX)
        return overlap_ratio * 0.6 + dist_factor * 0.4
    else:
        # Horizontal: ruby above base
        v_dist = base_bbox.y_min - ruby_bbox.y_max
        if v_dist < -RUBY_PROXIMITY_MAX_PX or v_dist > RUBY_PROXIMITY_MAX_PX:
            return 0.0
        # Horizontal overlap
        overlap_left = max(ruby_bbox.x_min, base_bbox.x_min)
        overlap_right = min(ruby_bbox.x_max, base_bbox.x_max)
        overlap = max(0, overlap_right - overlap_left)
        extent = min(ruby_bbox.width, base_bbox.width)
        if extent <= 0:
            return 0.0
        overlap_ratio = overlap / extent
        dist_factor = max(0, 1.0 - abs(v_dist) / RUBY_PROXIMITY_MAX_PX)
        return overlap_ratio * 0.6 + dist_factor * 0.4


# ---------------------------------------------------------------------------
# Ruby detection (per page)
# ---------------------------------------------------------------------------

def detect_ruby_candidates(
    tokens: list[OCRToken],
) -> tuple[RubyMode, float, list[RubyAttachment]]:
    """Analyse tokens to detect ruby annotations.

    Returns:
        (ruby_mode, confidence, attachments)
    """
    if len(tokens) < 2:
        return RubyMode.NONE, 0.0, []

    orientation = _classify_orientation(tokens)
    median_h, std_h = _compute_height_stats(tokens)
    if median_h <= 0:
        return RubyMode.NONE, 0.0, []

    # Separate small (ruby candidate) and large (base candidate) tokens
    small_tokens: list[OCRToken] = []
    base_tokens: list[OCRToken] = []
    for t in tokens:
        if _is_small_box(t, median_h):
            small_tokens.append(t)
            t.is_ruby_candidate = True
        else:
            base_tokens.append(t)

    if not small_tokens or not base_tokens:
        return RubyMode.NONE, 0.0, []

    # Compute overall ruby likelihood
    small_ratio = len(small_tokens) / len(tokens)

    # Try to match each small token to the best base token
    attachments: list[RubyAttachment] = []
    matched_count = 0

    for ruby_tok in small_tokens:
        best_score = 0.0
        best_base: Optional[OCRToken] = None

        for base_tok in base_tokens:
            score = _proximity_score(ruby_tok.bbox, base_tok.bbox, orientation)
            # Bonus for height ratio being in ideal range (0.25–0.45)
            h_ratio = ruby_tok.bbox.height / base_tok.bbox.height if base_tok.bbox.height > 0 else 0
            if 0.20 <= h_ratio <= 0.55:
                score *= 1.2
            if score > best_score:
                best_score = score
                best_base = base_tok

        if best_base is not None and best_score > 0.3:
            attachments.append(RubyAttachment(
                base_text=best_base.text,
                ruby_text=ruby_tok.text,
                base_bbox=best_base.bbox.to_list(),
                ruby_bbox=ruby_tok.bbox.to_list(),
                confidence=min(best_score, 1.0),
            ))
            matched_count += 1

    if not attachments:
        return RubyMode.NONE, 0.0, []

    # Confidence: combine match rate and small-box ratio
    match_rate = matched_count / len(small_tokens)
    confidence = 0.4 * match_rate + 0.3 * min(small_ratio * 5, 1.0) + 0.3 * (
        statistics.mean([a.confidence for a in attachments])
    )
    confidence = min(confidence, 1.0)

    if confidence >= RUBY_DETECTED_THRESHOLD:
        mode = RubyMode.DETECTED
    elif confidence >= RUBY_POSSIBLE_THRESHOLD:
        mode = RubyMode.POSSIBLE
    else:
        mode = RubyMode.NONE

    logger.debug(
        f"Ruby detection: {len(attachments)} attachments, "
        f"confidence={confidence:.2f}, mode={mode.value}"
    )
    return mode, confidence, attachments


# ---------------------------------------------------------------------------
# Text generation from attachments
# ---------------------------------------------------------------------------

def generate_ruby_text(plain: str, attachments: list[RubyAttachment]) -> str:
    """Generate annotated text like ``漢字(かんじ)``."""
    if not attachments:
        return plain
    result = plain
    # Sort attachments by descending base_text length to avoid partial replacements
    sorted_atts = sorted(attachments, key=lambda a: len(a.base_text), reverse=True)
    replaced: set[str] = set()
    for att in sorted_atts:
        if att.base_text in replaced:
            continue
        annotated = f"{att.base_text}({att.ruby_text})"
        # Replace only first occurrence
        result = result.replace(att.base_text, annotated, 1)
        replaced.add(att.base_text)
    return result


def generate_ruby_html(plain: str, attachments: list[RubyAttachment]) -> str:
    """Generate HTML with ``<ruby>`` tags."""
    if not attachments:
        return plain
    result = plain
    sorted_atts = sorted(attachments, key=lambda a: len(a.base_text), reverse=True)
    replaced: set[str] = set()
    for att in sorted_atts:
        if att.base_text in replaced:
            continue
        html_ruby = f"<ruby>{att.base_text}<rt>{att.ruby_text}</rt></ruby>"
        result = result.replace(att.base_text, html_ruby, 1)
        replaced.add(att.base_text)
    return result


# ---------------------------------------------------------------------------
# Build structured lines from tokens + attachments
# ---------------------------------------------------------------------------

def build_structured_lines(
    tokens: list[OCRToken],
    attachments: list[RubyAttachment],
    page_index: int,
) -> list[StructuredLine]:
    """Group tokens into lines and annotate with ruby segments."""
    if not tokens:
        return []

    # Build a lookup: base_text → ruby_text
    ruby_map: dict[str, str] = {}
    for att in attachments:
        ruby_map[att.base_text] = att.ruby_text

    # Group tokens by (block_order, line_order)
    line_groups: dict[tuple[int, int], list[OCRToken]] = {}
    for t in tokens:
        if t.is_ruby_candidate:
            continue  # skip ruby tokens from the line listing
        key = (t.block_order, t.line_order)
        line_groups.setdefault(key, []).append(t)

    lines: list[StructuredLine] = []
    for (blk, ln), group_tokens in sorted(line_groups.items()):
        group_tokens.sort(key=lambda t: t.token_order)
        orientation = _classify_orientation(group_tokens)
        line_id = f"{page_index:04d}-{blk:02d}-{ln:02d}"

        segments: list[LineSegment] = []
        for t in group_tokens:
            segments.append(LineSegment(seg_type="base", text=t.text))
            if t.text in ruby_map:
                segments.append(LineSegment(
                    seg_type="ruby",
                    text=ruby_map[t.text],
                    parent=t.text,
                ))

        lines.append(StructuredLine(
            line_id=line_id,
            orientation=orientation,
            segments=segments,
            block_order=blk,
            line_order=ln,
        ))

    return lines


# ---------------------------------------------------------------------------
# High-level: enrich an OCRPageResult with ruby data
# ---------------------------------------------------------------------------

def enrich_with_ruby(result: OCRPageResult) -> OCRPageResult:
    """Add ruby detection results to an existing ``OCRPageResult``.

    Mutates the result in-place and returns it.
    """
    mode, confidence, attachments = detect_ruby_candidates(result.tokens)

    result.ruby_detected = mode in (RubyMode.DETECTED, RubyMode.POSSIBLE)
    result.ruby_confidence = confidence
    result.ruby_mode = mode.value
    result.ruby_candidates_count = len(attachments)
    result.ruby_attachments = attachments

    if attachments:
        result.ruby_text = generate_ruby_text(result.plain_text, attachments)
        result.ruby_html = generate_ruby_html(result.plain_text, attachments)
    else:
        result.ruby_text = result.plain_text
        result.ruby_html = result.plain_text

    result.lines = build_structured_lines(
        result.tokens, attachments, result.page_index
    )

    return result
