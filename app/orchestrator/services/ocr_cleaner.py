"""OCR text cleaning: rule-based + LLM two-pass approach.

Pass 1 (rule-based): Removes safe, deterministic noise patterns.
Pass 2 (LLM): Identifies and removes ambiguous noise without altering novel content.

Uses llm_manager.acquire/release for automatic LLM lifecycle management.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import requests

from app.shared.logger import get_logger

logger = get_logger("ocr_cleaner")

# ── Rule-based patterns ──────────────────────────────────────────────────────

# OCR page markers inserted by ingest pipeline
_RE_OCR_MARKER = re.compile(r"^={3,}\s*OCR:\s*.+\s*={3,}$", re.MULTILINE)

# Page-number-only lines (1-6 digits, possibly with surrounding whitespace/dashes)
_RE_PAGE_NUMBER = re.compile(r"^\s*[-–—]?\s*\d{1,6}\s*[-–—]?\s*$", re.MULTILINE)

# Lines of consecutive symbols (≥5 non-letter/non-kana/non-kanji chars)
_RE_SYMBOL_NOISE = re.compile(
    r"^[^\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}a-zA-Z0-9\s]{5,}$",
    re.MULTILINE,
)

# Fallback: simpler ASCII symbol noise
_RE_ASCII_NOISE = re.compile(r"^[^a-zA-Z0-9\u3000-\u9fff\u30a0-\u30ff\u3040-\u309f\s]{5,}$", re.MULTILINE)

# Repeated separator lines (===, ---, ___, etc.)
_RE_SEPARATOR = re.compile(r"^[-=_~*#]{3,}\s*$", re.MULTILINE)

# Excessive blank lines (3+)
_RE_EXCESS_BLANKS = re.compile(r"\n{4,}")


@dataclass
class RemovedSpan:
    text: str
    reason: str
    line_number: Optional[int] = None


@dataclass
class CleaningResult:
    cleaned_text: str
    removed_spans: List[RemovedSpan] = field(default_factory=list)
    quality_flags: Dict[str, object] = field(default_factory=dict)


# ── Rule-based pass ──────────────────────────────────────────────────────────

def _rule_based_clean(text: str) -> tuple[str, List[RemovedSpan]]:
    """Remove safe, deterministic noise patterns. Returns (cleaned, removed)."""
    removed: List[RemovedSpan] = []
    lines = text.split("\n")
    cleaned_lines = []

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # OCR markers
        if _RE_OCR_MARKER.match(stripped):
            removed.append(RemovedSpan(text=stripped, reason="ocr_marker", line_number=i))
            continue

        # Page numbers only
        if _RE_PAGE_NUMBER.match(stripped) and len(stripped) < 10:
            removed.append(RemovedSpan(text=stripped, reason="page_number", line_number=i))
            continue

        # Separator lines
        if _RE_SEPARATOR.match(stripped):
            removed.append(RemovedSpan(text=stripped, reason="separator", line_number=i))
            continue

        # ASCII symbol noise
        if _RE_ASCII_NOISE.match(stripped):
            removed.append(RemovedSpan(text=stripped, reason="symbol_noise", line_number=i))
            continue

        cleaned_lines.append(line)

    result = "\n".join(cleaned_lines)
    # Collapse excessive blank lines
    result = _RE_EXCESS_BLANKS.sub("\n\n\n", result)

    return result, removed


# ── LLM pass ────────────────────────────────────────────────────────────────

_LLM_SYSTEM_PROMPT = """あなたはOCRテキストのノイズ除去アシスタントです。
以下のテキストから、OCR処理で混入したノイズ行のみを特定してください。

重要なルール:
- 小説の本文（台詞、地の文、心内語、描写）は絶対に削除しないでください
- 章見出しは保持してください
- 明らかなOCRゴミ（意味不明な文字列、壊れた文字の連続、OCR処理の残骸）のみ対象です
- 判断に迷う場合は保持してください（誤削除よりノイズ残留のほうが安全）

JSONで回答してください:
{"noise_lines": [{"line_number": N, "text": "該当テキスト", "reason": "理由"}]}

ノイズがない場合: {"noise_lines": []}"""


def _llm_clean_chunk(
    chunk: str,
    api_url: str,
    api_key: str = "",
    model: str = "",
) -> List[RemovedSpan]:
    """Send a text chunk to LLM for noise identification."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Number lines for LLM reference
    lines = chunk.split("\n")
    numbered = "\n".join(f"{i+1}: {line}" for i, line in enumerate(lines))

    payload = {
        "messages": [
            {"role": "system", "content": _LLM_SYSTEM_PROMPT},
            {"role": "user", "content": f"以下のOCRテキストのノイズ行を特定してください:\n\n{numbered}"},
        ],
        "temperature": 0.1,
        "max_tokens": 1000,
    }
    if model:
        payload["model"] = model

    try:
        url = api_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url = f"{url}/chat/completions"

        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        # Strip markdown code fences
        content = re.sub(r"^```(?:json)?\s*", "", content.strip())
        content = re.sub(r"\s*```$", "", content.strip())

        parsed = json.loads(content)
        noise_lines = parsed.get("noise_lines", [])

        removed = []
        for item in noise_lines:
            removed.append(RemovedSpan(
                text=item.get("text", ""),
                reason=f"llm:{item.get('reason', 'noise')}",
                line_number=item.get("line_number"),
            ))
        return removed

    except Exception as e:
        logger.warning(f"LLM noise detection failed: {e}")
        return []


def _apply_llm_removals(text: str, removals: List[RemovedSpan]) -> str:
    """Remove lines identified by LLM."""
    if not removals:
        return text

    lines = text.split("\n")
    remove_indices = set()
    for r in removals:
        if r.line_number and 1 <= r.line_number <= len(lines):
            # Verify the line text roughly matches
            actual = lines[r.line_number - 1].strip()
            expected = r.text.strip()
            if expected and (expected in actual or actual in expected):
                remove_indices.add(r.line_number - 1)

    cleaned = [line for i, line in enumerate(lines) if i not in remove_indices]
    return "\n".join(cleaned)


# ── Public API ───────────────────────────────────────────────────────────────

def clean_ocr_text_with_llm(
    text: str,
    mode: str = "novel",
    use_llm: bool = True,
    chunk_size: int = 3000,
) -> CleaningResult:
    """Clean OCR text using rule-based + optional LLM two-pass approach.

    Args:
        text: Raw OCR text to clean.
        mode: Cleaning mode ("novel" optimized for fiction).
        use_llm: Whether to use LLM for ambiguous noise (requires running LLM).
        chunk_size: Max characters per LLM chunk.

    Returns:
        CleaningResult with cleaned text, removed spans, and quality flags.
    """
    all_removed: List[RemovedSpan] = []

    # Pass 1: Rule-based
    cleaned, rule_removed = _rule_based_clean(text)
    all_removed.extend(rule_removed)
    logger.info(f"[ocr_cleaner] Rule-based pass removed {len(rule_removed)} spans")

    # Pass 2: LLM (optional)
    if use_llm:
        api_url = os.environ.get("LLM_API_URL", "")
        if api_url:
            try:
                from app.orchestrator.services.llm_manager import acquire, release, touch_last_used

                acquire("ocr_cleanup")
                try:
                    api_key = os.environ.get("LLM_API_KEY", "")
                    model = os.environ.get("LLM_MODEL", "")

                    # Process in chunks
                    chunks = _split_into_chunks(cleaned, chunk_size)
                    llm_removed_all: List[RemovedSpan] = []

                    for chunk in chunks:
                        llm_removed = _llm_clean_chunk(chunk, api_url, api_key, model)
                        llm_removed_all.extend(llm_removed)
                        touch_last_used()

                    if llm_removed_all:
                        cleaned = _apply_llm_removals(cleaned, llm_removed_all)
                        all_removed.extend(llm_removed_all)
                        logger.info(f"[ocr_cleaner] LLM pass removed {len(llm_removed_all)} spans")
                finally:
                    release("ocr_cleanup")

            except ImportError:
                logger.warning("[ocr_cleaner] llm_manager not available, skipping LLM pass")
        else:
            logger.info("[ocr_cleaner] LLM_API_URL not set, skipping LLM pass")

    # Quality flags
    total_chars = len(text)
    cleaned_chars = len(cleaned)
    removed_chars = total_chars - cleaned_chars

    # Estimate garbled ratio (non-standard character density)
    garbled_count = len(re.findall(r"[^\u3000-\u9fff\u30a0-\u30ff\u3040-\u309fa-zA-Z0-9\s。、！？「」『』（）…ー〜・]", cleaned))
    garbled_ratio = garbled_count / max(cleaned_chars, 1)

    quality_flags = {
        "garbled_ratio": round(garbled_ratio, 4),
        "has_noise": garbled_ratio > 0.05,
        "removed_chars": removed_chars,
        "removed_ratio": round(removed_chars / max(total_chars, 1), 4),
        "removed_spans_count": len(all_removed),
    }

    return CleaningResult(
        cleaned_text=cleaned,
        removed_spans=[{"text": r.text, "reason": r.reason, "line_number": r.line_number} for r in all_removed],
        quality_flags=quality_flags,
    )


def _split_into_chunks(text: str, max_chars: int) -> List[str]:
    """Split text into chunks at paragraph boundaries."""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    current = ""
    for para in text.split("\n\n"):
        if len(current) + len(para) + 2 > max_chars and current:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para

    if current:
        chunks.append(current)
    return chunks
