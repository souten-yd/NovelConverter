"""Layer 3 – Parallel OCR Executor.

Executes OCR on pages using the engine assigned by the scheduler.
Parallelisation strategy:
  - PaddleOCR (GPU): ThreadPoolExecutor (GIL-released during GPU inference)
  - NDLOCR-Lite (CPU): ProcessPoolExecutor (full CPU parallelism)

Every result carries its ``page_index`` so that downstream stages can
restore the original order regardless of completion order.
"""
from __future__ import annotations

import time
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

from app.shared.logger import get_logger
from app.orchestrator.services.ocr.models import (
    BBox,
    EngineType,
    OCRPageResult,
    OCRToken,
    PageJob,
    PipelineConfig,
    ScheduleDecision,
)

logger = get_logger("ocr.executor")


# ---------------------------------------------------------------------------
# Per-engine execution functions
# ---------------------------------------------------------------------------

def _run_paddle_ocr(
    image_path: str,
    page_index: int,
    use_layout: bool = False,
    device: str = "gpu:0",
    lang: str = "japan",
) -> OCRPageResult:
    """Run PaddleOCR on a single page.  Returns an OCRPageResult with tokens."""
    result = OCRPageResult(page_index=page_index, image_path=image_path)
    result.engine = EngineType.PADDLE_LAYOUT.value if use_layout else EngineType.PADDLE_FAST.value
    result.start_timer()

    try:
        from app.orchestrator.services.ocr.paddleocr_engine import PaddleOCREngine

        engine = PaddleOCREngine()
        available, missing = engine.is_available()
        if not available:
            result.status = "error"
            result.error_message = f"PaddleOCR unavailable: {missing}"
            result.stop_timer()
            return result

        text, warnings = engine.extract_text(Path(image_path), lang=f"{'jpn' if lang == 'japan' else 'eng'}+eng")
        result.warnings.extend(warnings)

        # Also get raw OCR data with bboxes
        _populate_tokens_from_paddle(result, image_path, lang)

        # Build plain text from tokens if available, otherwise use engine output
        if result.tokens:
            # Sort tokens by reading order and join
            sorted_tokens = sorted(
                result.tokens,
                key=lambda t: (t.block_order, t.line_order, t.token_order),
            )
            result.plain_text = "\n".join(t.text for t in sorted_tokens if t.text.strip())
        else:
            # Strip the "===== OCR: ... =====" header if present
            plain = text
            if plain.startswith("===== OCR:"):
                newline_pos = plain.find("\n")
                if newline_pos >= 0:
                    plain = plain[newline_pos + 1:]
            result.plain_text = plain.strip()

    except Exception as exc:
        result.status = "error"
        result.error_message = str(exc)
        logger.warning(f"PaddleOCR failed for page {page_index}: {exc}")

    result.stop_timer()
    return result


def _populate_tokens_from_paddle(result: OCRPageResult, image_path: str, lang: str) -> None:
    """Extract token-level bounding boxes from PaddleOCR raw output."""
    try:
        from app.orchestrator.services.ocr.paddleocr_engine import PaddleOCREngine

        ocr = PaddleOCREngine._ocr_instance
        if ocr is None:
            return

        raw = ocr.ocr(image_path, cls=True)
        if not raw or not raw[0]:
            return

        for line_idx, line_info in enumerate(raw[0]):
            if not line_info or len(line_info) < 2:
                continue
            bbox_points = line_info[0]
            text_info = line_info[1]
            text = text_info[0] if isinstance(text_info, (list, tuple)) else str(text_info)
            conf = text_info[1] if isinstance(text_info, (list, tuple)) and len(text_info) > 1 else 0.0

            bbox = BBox.from_polygon(bbox_points)
            result.tokens.append(OCRToken(
                text=text,
                bbox=bbox,
                confidence=float(conf),
                block_order=0,
                line_order=line_idx,
                token_order=0,
            ))
    except Exception as exc:
        logger.debug(f"Token extraction failed for page {result.page_index}: {exc}")


def _run_ndlocr_lite(
    image_path: str,
    page_index: int,
) -> OCRPageResult:
    """Run NDLOCR-Lite on a single page."""
    result = OCRPageResult(page_index=page_index, image_path=image_path)
    result.engine = EngineType.NDLOCR_LITE.value
    result.start_timer()

    try:
        from app.orchestrator.services.ocr.ndlocr_lite_engine import NDLOCRLiteEngine

        engine = NDLOCRLiteEngine()
        available, missing = engine.is_available()
        if not available:
            result.status = "error"
            result.error_message = f"NDLOCR-Lite unavailable: {missing}"
            result.stop_timer()
            return result

        text, warnings = engine.extract_text(Path(image_path))
        result.warnings.extend(warnings)

        # Strip the header
        plain = text
        if plain.startswith("===== OCR:"):
            newline_pos = plain.find("\n")
            if newline_pos >= 0:
                plain = plain[newline_pos + 1:]
        result.plain_text = plain.strip()

        # NDLOCR-Lite doesn't provide bboxes, so tokens stay empty
        # Ruby detection will be limited for this engine

    except Exception as exc:
        result.status = "error"
        result.error_message = str(exc)
        logger.warning(f"NDLOCR-Lite failed for page {page_index}: {exc}")

    result.stop_timer()
    return result


def _run_fallback(
    image_path: str,
    page_index: int,
) -> OCRPageResult:
    """Run Tesseract (fallback) on a single page."""
    result = OCRPageResult(page_index=page_index, image_path=image_path)
    result.engine = EngineType.FALLBACK.value
    result.start_timer()

    try:
        from app.orchestrator.services.ocr.factory import get_engine

        engine = get_engine("tesseract")
        available, missing = engine.is_available()
        if not available:
            result.status = "error"
            result.error_message = f"Tesseract unavailable: {missing}"
            result.stop_timer()
            return result

        text, warnings = engine.extract_text(Path(image_path))
        result.warnings.extend(warnings)
        plain = text
        if plain.startswith("===== OCR:"):
            newline_pos = plain.find("\n")
            if newline_pos >= 0:
                plain = plain[newline_pos + 1:]
        result.plain_text = plain.strip()

    except Exception as exc:
        result.status = "error"
        result.error_message = str(exc)
        logger.warning(f"Fallback OCR failed for page {page_index}: {exc}")

    result.stop_timer()
    return result


# ---------------------------------------------------------------------------
# Top-level NDLOCR worker function (must be picklable for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _ndlocr_worker(image_path: str, page_index: int) -> dict:
    """Process pool target for NDLOCR-Lite.  Returns a serializable dict."""
    result = _run_ndlocr_lite(image_path, page_index)
    return result.to_dict()


# ---------------------------------------------------------------------------
# Parallel executor
# ---------------------------------------------------------------------------

def execute_ocr(
    jobs: list[PageJob],
    decisions: list[ScheduleDecision],
    config: Optional[PipelineConfig] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> list[OCRPageResult]:
    """Execute OCR on all pages in parallel, preserving page_index metadata.

    PaddleOCR jobs run in a ThreadPoolExecutor (GPU-bound).
    NDLOCR-Lite jobs run in a ProcessPoolExecutor (CPU-bound).
    Fallback jobs run sequentially.

    Results are returned in *completion order* – the caller must sort by
    ``page_index`` to restore document order.

    Args:
        jobs: Page jobs from the normalizer.
        decisions: Routing decisions from the scheduler.
        config: Pipeline configuration.
        progress_cb: Optional callback(completed_count, total_count).

    Returns:
        List of OCRPageResult (unsorted – caller must sort by page_index).
    """
    if config is None:
        config = PipelineConfig()

    decision_map = {d.page_index: d for d in decisions}
    results: list[OCRPageResult] = []
    total = len(jobs)
    completed = 0

    # Group jobs by engine type
    paddle_jobs: list[PageJob] = []
    ndlocr_jobs: list[PageJob] = []
    fallback_jobs: list[PageJob] = []

    for job in jobs:
        dec = decision_map.get(job.page_index)
        if dec is None:
            fallback_jobs.append(job)
            continue
        if dec.engine in (EngineType.PADDLE_FAST, EngineType.PADDLE_LAYOUT):
            paddle_jobs.append(job)
        elif dec.engine == EngineType.NDLOCR_LITE:
            ndlocr_jobs.append(job)
        else:
            fallback_jobs.append(job)

    logger.info(
        f"Executor: {len(paddle_jobs)} paddle, {len(ndlocr_jobs)} ndlocr, "
        f"{len(fallback_jobs)} fallback ({total} total)"
    )

    futures: list[Future] = []

    # --- PaddleOCR (ThreadPool, GPU) ---
    if paddle_jobs:
        paddle_pool = ThreadPoolExecutor(
            max_workers=config.paddle_max_workers,
            thread_name_prefix="paddle",
        )
        for job in paddle_jobs:
            dec = decision_map[job.page_index]
            use_layout = dec.engine == EngineType.PADDLE_LAYOUT
            f = paddle_pool.submit(
                _run_paddle_ocr,
                job.image_path,
                job.page_index,
                use_layout=use_layout,
                device=config.paddle_device,
                lang=config.paddle_lang,
            )
            f._ocr_page_index = job.page_index  # type: ignore[attr-defined]
            futures.append(f)

    # --- NDLOCR-Lite (ProcessPool, CPU) ---
    ndlocr_pool: Optional[ProcessPoolExecutor] = None
    if ndlocr_jobs:
        # Use ProcessPoolExecutor for true CPU parallelism
        # NOTE: We use ThreadPoolExecutor as a safe fallback because NDLOCR-Lite
        # is CLI-based (subprocess) – it already spawns separate processes.
        # ProcessPoolExecutor would add unnecessary overhead for a CLI wrapper.
        ndlocr_pool = ThreadPoolExecutor(
            max_workers=config.ndlocr_max_workers,
            thread_name_prefix="ndlocr",
        )
        for job in ndlocr_jobs:
            f = ndlocr_pool.submit(
                _run_ndlocr_lite,
                job.image_path,
                job.page_index,
            )
            f._ocr_page_index = job.page_index  # type: ignore[attr-defined]
            futures.append(f)

    # Collect futures
    for future in as_completed(futures):
        try:
            res = future.result()
            if isinstance(res, dict):
                # From process pool worker – reconstruct
                page_result = _dict_to_page_result(res)
            else:
                page_result = res
            results.append(page_result)
        except Exception as exc:
            page_idx = getattr(future, "_ocr_page_index", -1)
            logger.error(f"OCR future failed for page {page_idx}: {exc}")
            results.append(OCRPageResult(
                page_index=page_idx,
                status="error",
                error_message=str(exc),
            ))

        completed += 1
        if progress_cb:
            try:
                progress_cb(completed, total)
            except Exception:
                pass

    # --- Fallback (sequential) ---
    for job in fallback_jobs:
        res = _run_fallback(job.image_path, job.page_index)
        results.append(res)
        completed += 1
        if progress_cb:
            try:
                progress_cb(completed, total)
            except Exception:
                pass

    # Shutdown pools
    if paddle_jobs:
        paddle_pool.shutdown(wait=False)
    if ndlocr_pool:
        ndlocr_pool.shutdown(wait=False)

    logger.info(f"Executor complete: {len(results)}/{total} pages processed")
    return results


def _dict_to_page_result(d: dict) -> OCRPageResult:
    """Reconstruct OCRPageResult from a serialized dict."""
    result = OCRPageResult(
        page_index=d.get("page_index", 0),
        engine=d.get("engine", ""),
        elapsed_ms=d.get("elapsed_ms", 0),
        plain_text=d.get("plain_text", ""),
        status=d.get("status", "ok"),
        error_message=d.get("error_message", ""),
        warnings=d.get("warnings", []),
        source_name=d.get("source_name", ""),
    )
    # Reconstruct tokens
    for td in d.get("tokens", []):
        result.tokens.append(OCRToken(
            text=td["text"],
            bbox=BBox.from_list(td["bbox"]),
            confidence=td.get("confidence", 0),
            block_order=td.get("block_order", 0),
            line_order=td.get("line_order", 0),
            token_order=td.get("token_order", 0),
            is_ruby_candidate=td.get("is_ruby_candidate", False),
        ))
    return result
