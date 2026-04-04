"""Thread-safe registry for ingest/upload progress tracking.

Each upload is identified by an ``upload_id`` (UUID string).
Progress data shape:
  {
    "stage":        str,   # see STAGE_* constants
    "stage_label":  str,   # human-readable Japanese label
    "page":         int,   # current page (0 if not paged)
    "total_pages":  int,   # total pages (0 if not known)
    "pct":          int,   # 0-100 completion estimate
    "warnings":     list[str],
    "status":       str,   # "running" | "complete" | "failed"
    "error":        str,   # non-empty on failure
  }
"""
from __future__ import annotations

import threading
import time
from typing import Optional

_store: dict[str, dict] = {}
_lock = threading.Lock()

# Stage constants (also used as SSE event 'stage' field)
STAGE_UPLOAD        = "upload"
STAGE_UNPACK        = "unpack"
STAGE_PAGE_SCAN     = "page_scan"
STAGE_OCR_PAGE      = "ocr_page"
STAGE_TEXT_MERGE    = "text_merge"
STAGE_NORMALIZE     = "normalize"
STAGE_SEGMENT_GEN   = "segment_generate"
STAGE_COMPLETE      = "complete"
STAGE_FAILED        = "failed"

_STAGE_LABELS: dict[str, str] = {
    STAGE_UPLOAD:       "アップロード受信中",
    STAGE_UNPACK:       "アーカイブ解凍中",
    STAGE_PAGE_SCAN:    "ページ検出中",
    STAGE_OCR_PAGE:     "OCR処理中",
    STAGE_TEXT_MERGE:   "テキスト結合中",
    STAGE_NORMALIZE:    "正規化中",
    STAGE_SEGMENT_GEN:  "セグメント生成中",
    STAGE_COMPLETE:     "完了",
    STAGE_FAILED:       "失敗",
}

# Rough percentage assignments per stage
_STAGE_PCT: dict[str, int] = {
    STAGE_UPLOAD:       5,
    STAGE_UNPACK:       15,
    STAGE_PAGE_SCAN:    20,
    STAGE_OCR_PAGE:     70,   # varies with page progress
    STAGE_TEXT_MERGE:   80,
    STAGE_NORMALIZE:    90,
    STAGE_SEGMENT_GEN:  95,
    STAGE_COMPLETE:     100,
    STAGE_FAILED:       0,
}


def create(upload_id: str) -> None:
    with _lock:
        _store[upload_id] = {
            "stage":        STAGE_UPLOAD,
            "stage_label":  _STAGE_LABELS[STAGE_UPLOAD],
            "page":         0,
            "total_pages":  0,
            "pct":          0,
            "warnings":     [],
            "status":       "running",
            "error":        "",
            "updated_at":   time.time(),
        }


def update(
    upload_id: str,
    stage: str,
    *,
    page: int = 0,
    total_pages: int = 0,
    warning: Optional[str] = None,
) -> None:
    with _lock:
        if upload_id not in _store:
            return
        d = _store[upload_id]
        d["stage"] = stage
        d["stage_label"] = _STAGE_LABELS.get(stage, stage)
        d["page"] = page
        d["total_pages"] = total_pages
        d["updated_at"] = time.time()

        # Calculate pct
        if stage == STAGE_OCR_PAGE and total_pages > 0:
            ocr_range_start = _STAGE_PCT[STAGE_UNPACK]
            ocr_range_end   = _STAGE_PCT[STAGE_TEXT_MERGE]
            frac = page / total_pages
            d["pct"] = int(ocr_range_start + frac * (ocr_range_end - ocr_range_start))
        else:
            d["pct"] = _STAGE_PCT.get(stage, d["pct"])

        if warning:
            d["warnings"].append(warning)


def complete(upload_id: str) -> None:
    with _lock:
        if upload_id not in _store:
            return
        d = _store[upload_id]
        d["stage"]       = STAGE_COMPLETE
        d["stage_label"] = _STAGE_LABELS[STAGE_COMPLETE]
        d["pct"]         = 100
        d["status"]      = "complete"
        d["updated_at"]  = time.time()


def fail(upload_id: str, error: str) -> None:
    with _lock:
        if upload_id not in _store:
            return
        d = _store[upload_id]
        d["stage"]       = STAGE_FAILED
        d["stage_label"] = _STAGE_LABELS[STAGE_FAILED]
        d["status"]      = "failed"
        d["error"]       = error
        d["updated_at"]  = time.time()


def get(upload_id: str) -> Optional[dict]:
    with _lock:
        data = _store.get(upload_id)
        return dict(data) if data else None


def cleanup(upload_id: str) -> None:
    """Remove progress entry (call after client has consumed final event)."""
    with _lock:
        _store.pop(upload_id, None)
