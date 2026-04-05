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
from app.orchestrator.services.ocr.numpy_safety import (
    deep_to_py_scalars,
    safe_float,
    safe_int,
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

    from app.orchestrator.services.resource_manager import acquire_lease, release_lease

    lease_reason = f"ocr_page_{page_index}"
    acquire_lease(
        "paddleocr",
        reason=lease_reason,
        options={
            "device": device,
            "use_layout": use_layout,
        },
    )
    try:
        from app.orchestrator.services.ocr.paddleocr_engine import PaddleOCREngine

        engine = PaddleOCREngine()
        PaddleOCREngine.configure(device=device, use_layout=use_layout)
        available, missing = engine.is_available()
        if not available:
            result.status = "error"
            result.error_message = f"PaddleOCR unavailable: {missing}"
            result.stop_timer()
            return result

        raw_items = engine.run_paddle_ocr(image_path, lang)
        if not raw_items:
            result.warnings.append(f"PaddleOCR returned empty result: {Path(image_path).name}")
        _populate_tokens_from_paddle(result, raw_items)

        # Build plain text from tokens if available, otherwise use engine output
        if result.tokens:
            # Sort tokens by reading order and join
            sorted_tokens = sorted(
                result.tokens,
                key=lambda t: (t.block_order, t.line_order, t.token_order),
            )
            result.plain_text = "\n".join(t.text for t in sorted_tokens if t.text.strip())
        else:
            result.plain_text = "\n".join(
                str(item.get("text", "")).strip()
                for item in raw_items
                if str(item.get("text", "")).strip()
            ).strip()

    except Exception as exc:
        result.status = "error"
        result.error_message = str(exc)
        logger.exception(
            "OCR pipeline failed: engine=%s stage=%s image_dtype=%s image_shape=%s bbox_type=%s score_type=%s page=%s",
            "paddleocr",
            "engine-call",
            result.debug.get("image_dtype", "unknown"),
            result.debug.get("image_shape", "unknown"),
            "unknown",
            "unknown",
            page_index,
        )
    finally:
        release_lease("paddleocr", reason=lease_reason)

    result.stop_timer()
    return result


def _populate_tokens_from_paddle(result: OCRPageResult, raw_items: list[dict]) -> None:
    """Extract token-level bounding boxes from PaddleOCR 3.x normalized output."""
    try:
        for line_idx, line_info in enumerate(raw_items):
            line_info = deep_to_py_scalars(line_info)
            text = str(line_info.get("text", "")).strip()
            if not text:
                continue
            conf = safe_float(line_info.get("score", 0.0)) or 0.0
            poly = line_info.get("poly")
            if poly is None:
                continue
            bbox = BBox.from_polygon(poly)
            result.tokens.append(OCRToken(
                text=text,
                bbox=bbox,
                confidence=conf,
                block_order=0,
                line_order=safe_int(line_idx) or 0,
                token_order=0,
            ))
        if result.tokens:
            t0 = result.tokens[0]
            logger.debug(
                "Paddle token sample: page=%s bbox_type=%s score_type=%s",
                result.page_index,
                type(t0.bbox.x_min).__name__,
                type(t0.confidence).__name__,
            )
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
        logger.exception(
            "OCR pipeline failed: engine=%s stage=%s image_dtype=%s image_shape=%s bbox_type=%s score_type=%s page=%s",
            "ndlocr_lite",
            "engine-call",
            "unknown",
            "unknown",
            "unknown",
            "unknown",
            page_index,
        )

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
        logger.exception(
            "OCR pipeline failed: engine=%s stage=%s image_dtype=%s image_shape=%s bbox_type=%s score_type=%s page=%s",
            "tesseract",
            "engine-call",
            "unknown",
            "unknown",
            "unknown",
            "unknown",
            page_index,
        )

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
        # Preflight once per job to avoid repeating the same API mismatch error
        try:
            from app.orchestrator.services.ocr.paddleocr_engine import PaddleOCREngine, _get_paddleocr_version

            preflight_engine = PaddleOCREngine()
            PaddleOCREngine.configure(device=config.paddle_device, use_layout=False)
            available, reason = preflight_engine.is_available()
            if available:
                preflight_engine._init_ocr(config.paddle_lang)
            else:
                raise RuntimeError(reason)
        except Exception as exc:
            version = "unknown"
            try:
                from app.orchestrator.services.ocr.paddleocr_engine import _get_paddleocr_version

                version = _get_paddleocr_version()
            except Exception:
                pass
            preflight_error = f"PaddleOCR preflight failed: {exc} (version={version})"
            logger.error(preflight_error)
            for job in paddle_jobs:
                fail = OCRPageResult(page_index=job.page_index, image_path=job.image_path)
                fail.engine = EngineType.PADDLE_FAST.value
                fail.status = "error"
                fail.error_message = preflight_error
                results.append(fail)
                completed += 1
                if progress_cb:
                    try:
                        progress_cb(completed, total)
                    except Exception:
                        pass
            paddle_jobs = []

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
    d = deep_to_py_scalars(d)
    result = OCRPageResult(
        page_index=safe_int(d.get("page_index", 0)) or 0,
        engine=d.get("engine", ""),
        elapsed_ms=safe_int(d.get("elapsed_ms", 0)) or 0,
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
            confidence=safe_float(td.get("confidence", 0)) or 0.0,
            block_order=safe_int(td.get("block_order", 0)) or 0,
            line_order=safe_int(td.get("line_order", 0)) or 0,
            token_order=safe_int(td.get("token_order", 0)) or 0,
            is_ruby_candidate=bool(td.get("is_ruby_candidate", False)),
        ))
    return result
