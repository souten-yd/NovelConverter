"""Layer 2 – OCR Scheduler.

Decides which OCR engine to route each page to, based on heuristic analysis
of the page image.  Also provides a preliminary ruby-likelihood estimate
by examining the image for small text regions.

Engine routing logic:
  - Simple text pages → ndlocr_lite  (CPU parallel, fast for clean text)
  - Complex layout / multi-column / figures → paddle_layout  (GPU, layout analysis)
  - Standard pages where PaddleOCR is preferred → paddle_fast  (GPU, no layout)
  - Pages with strong ruby indicators → paddle_layout  (better at small text)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from app.shared.logger import get_logger
from app.orchestrator.services.ocr.models import (
    EngineType,
    PageJob,
    PipelineConfig,
    RubyMode,
    ScheduleDecision,
)

logger = get_logger("ocr.scheduler")


# ---------------------------------------------------------------------------
# Image-level heuristics (lightweight, runs on CPU pre-OCR)
# ---------------------------------------------------------------------------

def _estimate_layout_complexity(image_path: str) -> float:
    """Estimate layout complexity from image features.

    Returns a score between 0 (very simple) and 1 (very complex).
    Uses edge density and contrast variance as proxies for complexity.
    """
    try:
        from PIL import Image
        import statistics

        img = Image.open(image_path).convert("L")
        w, h = img.size

        # Downsample for speed
        if w > 800:
            ratio = 800 / w
            img = img.resize((800, int(h * ratio)))
            w, h = img.size

        pixels = list(img.getdata())

        # Compute horizontal edge density (simple gradient)
        edge_count = 0
        total = 0
        for y in range(h):
            for x in range(1, w):
                idx = y * w + x
                diff = abs(pixels[idx] - pixels[idx - 1])
                if diff > 30:
                    edge_count += 1
                total += 1

        edge_density = edge_count / total if total > 0 else 0

        # Variance of pixel values (high variance → complex)
        mean_px = sum(pixels) / len(pixels)
        variance = sum((p - mean_px) ** 2 for p in pixels) / len(pixels)
        norm_var = min(variance / 5000, 1.0)

        # Combine
        complexity = 0.6 * min(edge_density * 5, 1.0) + 0.4 * norm_var
        return min(complexity, 1.0)

    except Exception as exc:
        logger.debug(f"Complexity estimation failed for {image_path}: {exc}")
        return 0.3  # neutral default


def _estimate_ruby_likelihood(image_path: str) -> float:
    """Quick estimate of whether the page might contain ruby text.

    Uses the distribution of connected-component sizes as a proxy:
    pages with ruby tend to have a bimodal size distribution (large base
    characters + small ruby characters).
    """
    try:
        from PIL import Image

        img = Image.open(image_path).convert("L")
        w, h = img.size

        # Downsample
        if w > 600:
            ratio = 600 / w
            img = img.resize((600, int(h * ratio)))
            w, h = img.size

        # Simple binarization
        threshold = 128
        pixels = list(img.getdata())
        binary = [1 if p < threshold else 0 for p in pixels]

        # Count dark pixel rows in vertical strips
        strip_w = w // 10
        dark_rows_per_strip: list[int] = []
        for s in range(10):
            x_start = s * strip_w
            x_end = min(x_start + strip_w, w)
            dark = 0
            for y in range(h):
                for x in range(x_start, x_end):
                    if binary[y * w + x]:
                        dark += 1
            dark_rows_per_strip.append(dark)

        if not dark_rows_per_strip:
            return 0.0

        # High variance across strips suggests mixed base+ruby columns
        mean_d = sum(dark_rows_per_strip) / len(dark_rows_per_strip)
        if mean_d == 0:
            return 0.0
        cv = (sum((d - mean_d) ** 2 for d in dark_rows_per_strip) / len(dark_rows_per_strip)) ** 0.5 / mean_d

        return min(cv * 0.8, 1.0)

    except Exception as exc:
        logger.debug(f"Ruby likelihood estimation failed for {image_path}: {exc}")
        return 0.0


# ---------------------------------------------------------------------------
# Engine availability check
# ---------------------------------------------------------------------------

def _is_engine_available(engine: EngineType) -> bool:
    """Check if an engine is available at runtime."""
    try:
        from app.orchestrator.services.ocr.factory import get_engine
        if engine == EngineType.NDLOCR_LITE:
            ok, _ = get_engine("ndlocr_lite").is_available()
            return ok
        if engine in (EngineType.PADDLE_FAST, EngineType.PADDLE_LAYOUT):
            ok, _ = get_engine("paddleocr").is_available()
            return ok
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def schedule_pages(
    jobs: list[PageJob],
    config: Optional[PipelineConfig] = None,
) -> list[ScheduleDecision]:
    """Assign an engine and ruby mode to each page.

    Args:
        jobs: Normalized page jobs with page_index.
        config: Pipeline configuration (thresholds, default engine).

    Returns:
        A ScheduleDecision per page, in page_index order.
    """
    if config is None:
        config = PipelineConfig()

    # Check which engines are available
    ndlocr_ok = _is_engine_available(EngineType.NDLOCR_LITE)
    paddle_ok = _is_engine_available(EngineType.PADDLE_FAST)

    decisions: list[ScheduleDecision] = []

    for job in jobs:
        complexity = _estimate_layout_complexity(job.image_path)
        ruby_likelihood = _estimate_ruby_likelihood(job.image_path) if config.enable_ruby_detection else 0.0

        # Routing decision
        if complexity >= config.complexity_threshold and paddle_ok:
            engine = EngineType.PADDLE_LAYOUT
        elif ruby_likelihood >= config.ruby_route_threshold and paddle_ok:
            engine = EngineType.PADDLE_LAYOUT
        elif config.default_engine == "paddle_fast" and paddle_ok:
            engine = EngineType.PADDLE_FAST
        elif config.default_engine == "paddle_layout" and paddle_ok:
            engine = EngineType.PADDLE_LAYOUT
        elif ndlocr_ok:
            engine = EngineType.NDLOCR_LITE
        elif paddle_ok:
            engine = EngineType.PADDLE_FAST
        else:
            engine = EngineType.FALLBACK

        # Ruby mode
        if ruby_likelihood >= 0.6:
            ruby_mode = RubyMode.AUTO_RUBY
        elif ruby_likelihood >= 0.3:
            ruby_mode = RubyMode.POSSIBLE
        else:
            ruby_mode = RubyMode.NONE

        decisions.append(ScheduleDecision(
            page_index=job.page_index,
            engine=engine,
            ruby_mode=ruby_mode,
            ruby_confidence=ruby_likelihood,
            layout_complexity=complexity,
        ))

        logger.debug(
            f"Page {job.page_index}: engine={engine.value} "
            f"ruby={ruby_mode.value}({ruby_likelihood:.2f}) "
            f"complexity={complexity:.2f}"
        )

    return decisions
