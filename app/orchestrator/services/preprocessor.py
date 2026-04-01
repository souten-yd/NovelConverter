"""Text preprocessing: clean, normalise, chapter detection, segment splitting."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

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


@dataclass
class RawSegment:
    chapter_index: int
    order_index: int
    text: str
    is_chapter_header: bool = False


# ── Public API ────────────────────────────────────────────────────────────────

def preprocess(raw_text: str) -> List[RawSegment]:
    """Full preprocessing pipeline. Returns ordered RawSegment list."""
    text = _normalize_whitespace(raw_text)
    chapters = _split_chapters(text)
    segments: List[RawSegment] = []
    order = 0
    for chap_idx, (header, body) in enumerate(chapters):
        if header:
            segments.append(RawSegment(chap_idx, order, header, is_chapter_header=True))
            order += 1
        para_segments = _split_body(body, chap_idx, order)
        segments.extend(para_segments)
        order += len(para_segments)
    logger.info(f"Preprocessed → {len(segments)} segments across {len(chapters)} chapter(s)")
    return segments


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
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    segments: List[RawSegment] = []
    order = start_order

    for para in paragraphs:
        sub = _split_paragraph(para)
        for s in sub:
            if s:
                segments.append(RawSegment(chap_idx, order, s))
                order += 1
    return segments


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
