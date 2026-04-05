"""Main OCR pipeline orchestrator.

Ties together the four layers:
  1. Input normalization  → PageJob list
  2. Scheduling           → ScheduleDecision per page
  3. Parallel execution   → OCRPageResult per page
  4. Post-processing      → sorted, ruby-enriched, formatted results

Usage::

    from app.orchestrator.services.ocr.pipeline import run_pipeline, PipelineConfig

    result = run_pipeline(
        input_path=Path("book.zip"),
        config=PipelineConfig(default_engine="ndlocr_lite"),
    )
    for page in result.pages:
        print(f"Page {page.page_index}: {page.plain_text[:80]}")
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Callable, Optional

from app.shared.logger import get_logger
from app.orchestrator.services.ocr.cache import OCRCache
from app.orchestrator.services.ocr.executor import execute_ocr
from app.orchestrator.services.ocr.input_normalizer import normalize_inputs, normalize_image_list
from app.orchestrator.services.ocr.models import (
    OCRPageResult,
    PageJob,
    PipelineConfig,
    PipelineResult,
)
from app.orchestrator.services.ocr.postprocessor import (
    compute_engine_stats,
    generate_combined_plain_text,
    postprocess,
)
from app.orchestrator.services.ocr.scheduler import schedule_pages

logger = get_logger("ocr.pipeline")


def run_pipeline(
    input_path: Optional[Path] = None,
    *,
    image_paths: Optional[list[Path]] = None,
    config: Optional[PipelineConfig] = None,
    job_id: Optional[str] = None,
    progress_cb: Optional[Callable[[str, int, int, int], None]] = None,
    page_index_overrides: Optional[list[int]] = None,
) -> PipelineResult:
    """Execute the full OCR pipeline on an input source.

    Provide either ``input_path`` (PDF/ZIP/folder/image) or ``image_paths``
    (pre-extracted list of image files).

    Args:
        input_path: Path to input file or directory.
        image_paths: Pre-extracted list of image file paths.
        config: Pipeline configuration.
        job_id: Unique job identifier (auto-generated if None).
        progress_cb: Optional callback(stage, current, total, pct).
        page_index_overrides: Optional page index mapping for subset re-runs.

    Returns:
        ``PipelineResult`` with all pages sorted by page_index.
    """
    if config is None:
        config = PipelineConfig()
    if job_id is None:
        job_id = f"job-{uuid.uuid4().hex[:8]}"

    start_time = time.monotonic()
    cache = OCRCache(cache_dir=config.cache_dir if config.cache_dir else None)
    cache.enabled = config.enable_cache

    def _cb(stage: str, cur: int, total: int, pct: int) -> None:
        if progress_cb:
            try:
                progress_cb(stage, cur, total, pct)
            except Exception:
                pass

    # ── Layer 1: Input normalization ──────────────────────────────────────
    _cb("normalizing", 0, 0, 5)
    logger.info(f"Pipeline start: job={job_id}")

    if image_paths is not None:
        jobs = normalize_image_list(
            image_paths,
            job_id=job_id,
            source_name=image_paths[0].parent.name if image_paths else "images",
        )
    elif input_path is not None:
        jobs = normalize_inputs(input_path, job_id=job_id)
    else:
        raise ValueError("Provide either input_path or image_paths")

    total_pages = len(jobs)
    if total_pages == 0:
        logger.warning("Pipeline: no pages to process")
        return PipelineResult(
            job_id=job_id,
            total_pages=0,
            total_elapsed_ms=0,
        )

    _cb("normalizing", total_pages, total_pages, 10)
    logger.info(f"Normalized: {total_pages} pages")

    # ── Layer 2: Scheduling ───────────────────────────────────────────────
    _cb("scheduling", 0, total_pages, 15)
    decisions = schedule_pages(jobs, config)
    _cb("scheduling", total_pages, total_pages, 20)

    # ── Check cache ───────────────────────────────────────────────────────
    cached_results: list[OCRPageResult] = []
    uncached_jobs: list[PageJob] = []
    uncached_decisions = []

    for job, dec in zip(jobs, decisions):
        if cache.enabled:
            cached = cache.get(job.image_path, dec.engine.value, dec.ruby_mode.value)
            if cached is not None:
                from app.orchestrator.services.ocr.executor import _dict_to_page_result
                page_result = _dict_to_page_result(cached)
                page_result.source_name = job.source_name
                cached_results.append(page_result)
                continue
        uncached_jobs.append(job)
        uncached_decisions.append(dec)

    if cached_results:
        logger.info(f"Cache: {len(cached_results)} hits, {len(uncached_jobs)} to process")

    # ── Layer 3: OCR Execution ────────────────────────────────────────────
    def _exec_progress(done: int, total: int) -> None:
        pct = 20 + int(60 * done / max(total, 1))
        _cb("ocr", done, total, pct)

    if uncached_jobs:
        ocr_results = execute_ocr(
            uncached_jobs,
            uncached_decisions,
            config=config,
            progress_cb=_exec_progress,
        )

        # Cache results
        decision_map = {d.page_index: d for d in uncached_decisions}
        for r in ocr_results:
            if r.status == "ok" and cache.enabled:
                dec = decision_map.get(r.page_index)
                if dec:
                    cache.put(
                        r.image_path,
                        dec.engine.value,
                        dec.ruby_mode.value,
                        r.to_dict(),
                    )
    else:
        ocr_results = []

    all_results = cached_results + ocr_results

    # ── Layer 4: Post-processing ──────────────────────────────────────────
    _cb("postprocessing", 0, total_pages, 85)
    all_results = postprocess(all_results, decisions, config)
    _cb("postprocessing", total_pages, total_pages, 95)

    if page_index_overrides is not None:
        if len(page_index_overrides) != len(all_results):
            logger.warning(
                "page_index_overrides length mismatch: overrides=%d results=%d",
                len(page_index_overrides), len(all_results),
            )
        for i, page in enumerate(all_results):
            if i < len(page_index_overrides):
                page.page_index = page_index_overrides[i]
        all_results.sort(key=lambda p: p.page_index)

    # Compute stats
    engine_stats = compute_engine_stats(all_results)
    total_ms = int((time.monotonic() - start_time) * 1000)

    pipeline_result = PipelineResult(
        job_id=job_id,
        pages=all_results,
        total_elapsed_ms=total_ms,
        total_pages=total_pages,
        engine_stats=engine_stats,
    )

    _cb("complete", total_pages, total_pages, 100)
    logger.info(
        f"Pipeline complete: job={job_id} pages={total_pages} "
        f"elapsed={total_ms}ms engines={engine_stats}"
    )

    return pipeline_result


def run_pipeline_for_ingest(
    image_paths: list[Path],
    *,
    ocr_engine: str = "tesseract",
    job_id: Optional[str] = None,
    progress_cb: Optional[Callable[[str, int, int, int], None]] = None,
) -> PipelineResult:
    """Convenience wrapper for the existing ingest flow.

    Maps the legacy ``ocr_engine`` setting to a ``PipelineConfig`` and
    runs the full pipeline on a list of image paths.
    """
    config = PipelineConfig()

    # Map legacy engine names to pipeline config
    if ocr_engine == "paddleocr":
        config.default_engine = "paddle_fast"
    elif ocr_engine == "ndlocr_lite":
        config.default_engine = "ndlocr_lite"
    elif ocr_engine == "tesseract":
        # Use fallback mode (Tesseract)
        config.default_engine = "fallback"
        config.enable_ruby_detection = False
    else:
        config.default_engine = "ndlocr_lite"

    return run_pipeline(
        image_paths=image_paths,
        config=config,
        job_id=job_id,
        progress_cb=progress_cb,
    )
