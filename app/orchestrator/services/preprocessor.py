"""Text preprocessing: clean, normalise, chapter detection, segment splitting."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from app.shared.logger import get_logger

logger = get_logger("preprocessor")

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_SEGMENT_CHARS = 200   # re-split if a segment exceeds this
CHAPTER_PATTERNS = [
    re.compile(r"^(第[一二三四五六七八九十百千\d]+[章節话話回][\s　]*.*)$", re.MULTILINE),
    re.compile(r"^(Chapter\s+\d+.*)$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^(【.+?】)$", re.MULTILINE),
    re.compile(r"^(\[.+?\])$", re.MULTILINE),
    re.compile(r"^(={3,}.*)$", re.MULTILINE),
    re.compile(r"^(-{3,}.*)$", re.MULTILINE),
    re.compile(r"^(■.+)$", re.MULTILINE),
    re.compile(r"^(◆.+)$", re.MULTILINE),
]

DIALOGUE_OPEN = re.compile(r"[「『]")
THOUGHT_OPEN = re.compile(r"（|[(]")  # parenthetical thought

SENTENCE_END = re.compile(r"([。！？!?…]+|[\r\n])")

# Monologue / inner-thought patterns (bracket-free)
_MONOLOGUE_PATTERNS = [
    re.compile(r"(〜?と思った|〜?と考えた|〜?と感じた|心の中で|心中で|胸の内で)"),
    re.compile(r"(と心の中で|心の奥で|頭の中で|脳裏に|胸の中で)"),
    re.compile(r"(と思う|と考える|と感じる|と悟った|と気づいた)"),
]


@dataclass
class RawSegment:
    chapter_index: int
    order_index: int
    text: str
    is_chapter_header: bool = False
    monologue_hint: Optional[str] = None  # "inner" | None
    rule_fired: str = ""                  # which rule produced this split


def _detect_monologue_hint(text: str) -> Optional[str]:
    """Return "inner" if the text contains a bracket-free inner-thought pattern."""
    for pat in _MONOLOGUE_PATTERNS:
        if pat.search(text):
            return "inner"
    return None


# ── Public API ────────────────────────────────────────────────────────────────

def preprocess(
    raw_text: str,
    normalizer_fn: Optional[Callable[[str], Tuple[str, list]]] = None,
    clean_with_llm: bool = False,
) -> Tuple[List[RawSegment], list]:
    """Full preprocessing pipeline. Returns (segments, normalization_log_entries).

    normalizer_fn: optional callable(str) -> (normalized_str, list[NormalizationEntry]).
    When not supplied the legacy _normalize_whitespace() is used and no log is produced.

    clean_with_llm: if True, runs OCR noise cleaning (rule-based + LLM) before
    chapter splitting. Requires LLM to be available for the LLM pass.
    """
    raw_chars = len(raw_text)
    raw_lines = raw_text.count("\n") + (1 if raw_text else 0)
    trimmed_chars = len(raw_text.strip())
    logger.info(
        f"Preprocess stage[input]: chars={raw_chars}, lines={raw_lines}, trimmed_chars={trimmed_chars}"
    )

    # OCR cleaning (rule-based always, LLM optional)
    ocr_cleaning_result = None
    try:
        from app.orchestrator.services.ocr_cleaner import clean_ocr_text_with_llm
        ocr_cleaning_result = clean_ocr_text_with_llm(
            raw_text, mode="novel", use_llm=clean_with_llm,
        )
        raw_text = ocr_cleaning_result.cleaned_text
        if ocr_cleaning_result.removed_spans:
            logger.info(
                f"Preprocess stage[ocr_clean]: removed {len(ocr_cleaning_result.removed_spans)} spans, "
                f"quality={ocr_cleaning_result.quality_flags}"
            )
    except Exception as e:
        logger.warning(f"Preprocess stage[ocr_clean]: skipped due to error: {e}")

    norm_logs: list = []
    if normalizer_fn is not None:
        text, norm_logs = normalizer_fn(raw_text)
    else:
        text = _normalize_whitespace(raw_text)

    logger.info(f"Preprocess stage[normalize]: chars={len(text)}")
    chapters = _split_chapters(text)
    logger.info(f"Preprocess stage[chapter_split]: chapters={len(chapters)}")
    segments: List[RawSegment] = []
    order = 0
    for chap_idx, (header, body) in enumerate(chapters):
        if header:
            segments.append(RawSegment(
                chap_idx, order, header,
                is_chapter_header=True,
                rule_fired="chapter_header",
            ))
            order += 1
        para_segments = _split_body(body, chap_idx, order)
        segments.extend(para_segments)
        order += len(para_segments)

    if not segments:
        logger.warning("Preprocess stage[segment_split]: 0 segments, activating paragraph fallback")
        fallback_segments = _fallback_split(text)
        for s in fallback_segments:
            segments.append(RawSegment(0, order, s, rule_fired="fallback"))
            order += 1
        logger.info(f"Preprocess stage[fallback]: segments={len(fallback_segments)}")

    # Attach monologue hints
    for seg in segments:
        if not seg.is_chapter_header:
            seg.monologue_hint = _detect_monologue_hint(seg.text)

    logger.info(f"Preprocessed → {len(segments)} segments across {len(chapters)} chapter(s)")
    return segments, norm_logs


# ── Internal helpers ──────────────────────────────────────────────────────────

def _normalize_whitespace(text: str) -> str:
    """Normalise line endings, compress blank lines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # full-width space to regular
    text = text.replace("\u3000", "　")
    # collapse 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_chapter_header(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    for pat in CHAPTER_PATTERNS:
        if pat.match(line):
            return True
    return False


def _split_chapters(text: str) -> List[tuple[Optional[str], str]]:
    """Split text into (header, body) pairs. Always at least one entry."""
    lines = text.split("\n")
    chapters: List[tuple[Optional[str], str]] = []
    current_header: Optional[str] = None
    current_lines: List[str] = []

    for line in lines:
        if _is_chapter_header(line):
            # flush current
            if current_lines or current_header is not None:
                chapters.append((current_header, "\n".join(current_lines).strip()))
            current_header = line.strip()
            current_lines = []
        else:
            current_lines.append(line)

    chapters.append((current_header, "\n".join(current_lines).strip()))
    if not chapters:
        chapters = [(None, text)]
    return chapters


def _split_body(body: str, chap_idx: int, start_order: int) -> List[RawSegment]:
    """Split chapter body into segments."""
    # First, join multi-line dialogue blocks into single paragraph units
    body = _join_multiline_dialogue(body)
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    segments: List[RawSegment] = []
    order = start_order

    for para in paragraphs:
        sub = _split_paragraph(para)
        for s in sub:
            if s:
                segments.append(RawSegment(chap_idx, order, s, rule_fired="paragraph"))
                order += 1
    return segments


def _join_multiline_dialogue(text: str) -> str:
    """Join 「...\\n...」 that spans multiple lines into a single line.

    Preserves the paragraph double-newline boundary so that paragraph
    splitting still works correctly after this pass.
    """
    lines = text.split("\n")
    result: List[str] = []
    buffer: Optional[str] = None   # accumulates an open-bracket block

    for line in lines:
        stripped = line.strip()
        if buffer is not None:
            # Inside an open dialogue block – append
            buffer += stripped
            # Check if it closed
            open_count = buffer.count("「") + buffer.count("『")
            close_count = buffer.count("」") + buffer.count("』")
            if close_count >= open_count:
                result.append(buffer)
                buffer = None
        else:
            # Count unclosed brackets in this line
            open_count = stripped.count("「") + stripped.count("『")
            close_count = stripped.count("」") + stripped.count("』")
            if open_count > close_count and stripped:
                # Start accumulating
                buffer = stripped
            else:
                result.append(line)

    if buffer is not None:
        result.append(buffer)

    return "\n".join(result)


def _split_paragraph(para: str) -> List[str]:
    """Split a paragraph at sentence boundaries if too long."""
    if len(para) <= MAX_SEGMENT_CHARS:
        return [para]

    # Split on sentence-ending punctuation
    parts: List[str] = []
    current = ""
    i = 0
    while i < len(para):
        c = para[i]
        current += c
        # dialogue block: keep together
        if c in "「『":
            depth = 1
            i += 1
            while i < len(para) and depth > 0:
                cc = para[i]
                current += cc
                if cc in "「『":
                    depth += 1
                elif cc in "」』":
                    depth -= 1
                i += 1
            # after closing bracket – natural split point
            if current.strip():
                parts.append(current.strip())
                current = ""
            continue
        if c in "。！？!?" and len(current) >= 10:
            parts.append(current.strip())
            current = ""
        i += 1

    if current.strip():
        if parts:
            # append to last if short
            if len(current.strip()) < 20:
                parts[-1] = parts[-1] + current.strip()
            else:
                parts.append(current.strip())
        else:
            parts.append(current.strip())

    # If still too long, hard-split
    result: List[str] = []
    for p in parts:
        if len(p) > MAX_SEGMENT_CHARS * 2:
            for chunk in _hard_split(p, MAX_SEGMENT_CHARS):
                result.append(chunk)
        else:
            result.append(p)
    return [r for r in result if r]


def _hard_split(text: str, size: int) -> List[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def _fallback_split(text: str) -> List[str]:
    """Fallback splitter to avoid zero segments when parser output is empty."""
    if not text.strip():
        return []
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if paras:
        return paras
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        return lines
    return [text.strip()]
