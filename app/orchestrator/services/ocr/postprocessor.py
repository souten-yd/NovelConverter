"""Layer 4 – Post-processing and integration.

Responsibilities:
  - Stable-sort all OCR results by page_index
  - Restore reading order within each page
  - Invoke ruby detection and attachment
  - Generate three output formats: plain text, ruby text, ruby HTML
  - Collect per-engine timing statistics
"""
from __future__ import annotations

from typing import Optional

from app.shared.logger import get_logger
from app.orchestrator.services.ocr.models import (
    EngineType,
    OCRPageResult,
    PipelineConfig,
    ScheduleDecision,
)
from app.orchestrator.services.ocr.ruby_detector import enrich_with_ruby

logger = get_logger("ocr.postprocessor")


# ---------------------------------------------------------------------------
# Reading order restoration
# ---------------------------------------------------------------------------

def _restore_reading_order(result: OCRPageResult) -> None:
    """Re-order tokens within a page to produce correct reading order.

    For PaddleOCR layout results we trust the engine's block ordering.
    For other engines we fall back to a geometric heuristic:
      - Group tokens into rows (horizontal) or columns (vertical)
      - Sort rows top-to-bottom, tokens left-to-right within a row
      - For vertical text: sort columns right-to-left, tokens top-to-bottom
    """
    if not result.tokens:
        return

    # If the engine already provides block/line ordering, trust it
    has_ordering = any(t.block_order > 0 or t.line_order > 0 for t in result.tokens)
    if has_ordering and result.engine in (
        EngineType.PADDLE_LAYOUT.value,
        EngineType.PADDLE_FAST.value,
    ):
        return

    # Detect dominant orientation
    vertical_count = sum(1 for t in result.tokens if t.bbox.height > t.bbox.width * 1.3)
    is_vertical = vertical_count > len(result.tokens) * 0.5

    if is_vertical:
        # Vertical text: sort columns right-to-left, then top-to-bottom within column
        # Group into columns by x-center proximity
        result.tokens.sort(key=lambda t: (-t.bbox.center[0], t.bbox.center[1]))
        _assign_block_line_orders(result.tokens, axis="x", reverse=True)
    else:
        # Horizontal text: sort rows top-to-bottom, then left-to-right within row
        result.tokens.sort(key=lambda t: (t.bbox.center[1], t.bbox.center[0]))
        _assign_block_line_orders(result.tokens, axis="y", reverse=False)


def _assign_block_line_orders(tokens: list, axis: str, reverse: bool) -> None:
    """Group tokens into lines and assign block_order / line_order.

    Tokens that are close together along the grouping axis are placed in
    the same line.  A gap larger than half the median token height starts
    a new line.
    """
    if not tokens:
        return

    heights = [t.bbox.height for t in tokens]
    median_h = sorted(heights)[len(heights) // 2] if heights else 20
    threshold = max(median_h * 0.5, 10)

    current_line = 0
    prev_val = None

    for idx, token in enumerate(tokens):
        if axis == "y":
            val = token.bbox.center[1]
        else:
            val = token.bbox.center[0]

        if prev_val is not None and abs(val - prev_val) > threshold:
            current_line += 1

        token.block_order = 0
        token.line_order = current_line
        token.token_order = idx
        prev_val = val


# ---------------------------------------------------------------------------
# Postprocess all pages
# ---------------------------------------------------------------------------

def postprocess(
    results: list[OCRPageResult],
    decisions: list[ScheduleDecision],
    config: Optional[PipelineConfig] = None,
) -> list[OCRPageResult]:
    """Post-process OCR results: order, ruby, and format generation.

    1. Stable-sort by page_index
    2. Restore reading order within each page
    3. Run ruby detection and enrichment
    4. Compute engine statistics

    Args:
        results: Raw OCR results (may be in any order).
        decisions: Scheduling decisions (for layout complexity data).
        config: Pipeline configuration.

    Returns:
        Results sorted by page_index with ruby data and formatted text.
    """
    if config is None:
        config = PipelineConfig()

    decision_map = {d.page_index: d for d in decisions}

    # 1. Stable sort by page_index
    results.sort(key=lambda r: r.page_index)

    for result in results:
        if result.status == "error":
            continue

        # Copy layout complexity from scheduling decision
        dec = decision_map.get(result.page_index)
        if dec:
            result.layout_complexity = dec.layout_complexity

        # 2. Restore reading order
        _restore_reading_order(result)

        # 3. Ruby detection and enrichment
        if config.enable_ruby_detection:
            enrich_with_ruby(result)
        else:
            result.ruby_text = result.plain_text
            result.ruby_html = result.plain_text

    logger.info(f"Post-processed {len(results)} pages")
    return results


def compute_engine_stats(results: list[OCRPageResult]) -> dict:
    """Compute per-engine statistics from results."""
    stats: dict = {}
    for r in results:
        engine = r.engine or "unknown"
        if engine not in stats:
            stats[engine] = {
                "count": 0,
                "total_ms": 0,
                "errors": 0,
                "avg_ms": 0,
            }
        stats[engine]["count"] += 1
        stats[engine]["total_ms"] += r.elapsed_ms
        if r.status == "error":
            stats[engine]["errors"] += 1

    for engine, s in stats.items():
        if s["count"] > 0:
            s["avg_ms"] = round(s["total_ms"] / s["count"])

    return stats


def generate_combined_plain_text(results: list[OCRPageResult]) -> str:
    """Generate combined plain text from all pages (sorted by page_index)."""
    parts: list[str] = []
    for r in sorted(results, key=lambda x: x.page_index):
        if r.plain_text:
            parts.append(r.plain_text)
    return "\n\n".join(parts)
