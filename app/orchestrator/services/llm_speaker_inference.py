"""LLM-primary chapter-unit speaker diarization.

Replaces the simple rule-based approach with LLM-driven speaker inference
that processes text chapter by chapter for better context utilization.

Phases:
  A. Text normalization and chapter splitting
  B. Utterance extraction (dialogue, thought, narration)
  C. LLM speaker inference per chapter
  D. Intra-chapter reconciliation
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable

import requests

from app.shared.logger import get_logger

logger = get_logger("llm_speaker_inference")

# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class Utterance:
    utterance_id: str                # "ch03_u014"
    chapter_index: int
    order_index: int
    text: str
    utterance_type: str              # dialogue / thought / narration / monologue
    surrounding_narration: str = ""  # nearby narration for context
    speaker: Optional[str] = None
    speaker_label: Optional[str] = None  # "未確定話者A" for unknowns
    confidence: float = 0.0
    reason: str = ""
    candidates: List[dict] = field(default_factory=list)
    needs_review: bool = False


@dataclass
class ChapterContext:
    chapter_index: int
    title: str
    text: str
    utterances: List[Utterance] = field(default_factory=list)
    characters_mentioned: List[str] = field(default_factory=list)


# ── Chapter heading patterns (reused from preprocessor) ─────────────────────

_CHAPTER_PATTERNS = [
    re.compile(r"^第[一二三四五六七八九十百千万\d]+[章節話回篇編幕]", re.MULTILINE),
    re.compile(r"^Chapter\s+\d+", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^【.+?】\s*$", re.MULTILINE),
    re.compile(r"^〈.+?〉\s*$", re.MULTILINE),
    re.compile(r"^プロローグ|^エピローグ", re.MULTILINE),
    re.compile(r"^Prologue|^Epilogue", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^■\s*.+", re.MULTILINE),
    re.compile(r"^◆\s*.+", re.MULTILINE),
]

# Bracket patterns for utterance extraction
_RE_DIALOGUE = re.compile(r"「([^」]+)」")
_RE_DIALOGUE_DOUBLE = re.compile(r"『([^』]+)』")
_RE_THOUGHT = re.compile(r"（([^）]+)）")

# Character name extraction from narration
_RE_SPEAKER_VERB = re.compile(
    r"([\u3000-\u9fff\u30a0-\u30ff]{1,10})[はがも](?:言|答|叫|呟|囁|尋|問|告|返|笑|怒|泣)"
)


# ── Phase A: Chapter splitting ───────────────────────────────────────────────

def split_into_chapters(text: str) -> List[ChapterContext]:
    """Split text into chapter-level chunks."""
    # Find all chapter heading positions
    headings: List[tuple[int, str]] = []
    for pattern in _CHAPTER_PATTERNS:
        for m in pattern.finditer(text):
            headings.append((m.start(), m.group().strip()))

    headings.sort(key=lambda x: x[0])

    if not headings:
        # No chapter headings found - treat entire text as one chapter
        return [ChapterContext(chapter_index=0, title="(全文)", text=text)]

    chapters = []
    for i, (pos, title) in enumerate(headings):
        end = headings[i + 1][0] if i + 1 < len(headings) else len(text)
        chapter_text = text[pos:end].strip()
        chapters.append(ChapterContext(
            chapter_index=i,
            title=title,
            text=chapter_text,
        ))

    # Include any text before the first chapter heading
    if headings[0][0] > 0:
        preamble = text[:headings[0][0]].strip()
        if preamble and len(preamble) > 50:
            chapters.insert(0, ChapterContext(
                chapter_index=-1,
                title="(前文)",
                text=preamble,
            ))
            # Re-index
            for i, ch in enumerate(chapters):
                ch.chapter_index = i

    return chapters


# ── Phase B: Utterance extraction ────────────────────────────────────────────

def extract_utterances(chapter: ChapterContext) -> List[Utterance]:
    """Extract dialogue, thought, and narration segments from chapter text."""
    text = chapter.text
    utterances: List[Utterance] = []
    idx = 0

    # Split into paragraphs
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    for para_idx, para in enumerate(paragraphs):
        # Skip chapter heading (first line often is the heading)
        if para_idx == 0 and any(p.match(para) for p in _CHAPTER_PATTERNS):
            continue

        # Find surrounding narration (previous and next non-dialogue paragraphs)
        surround_parts = []
        for offset in [-2, -1, 1, 2]:
            si = para_idx + offset
            if 0 <= si < len(paragraphs):
                candidate = paragraphs[si]
                if not _RE_DIALOGUE.search(candidate) and not _RE_DIALOGUE_DOUBLE.search(candidate):
                    surround_parts.append(candidate[:100])
        surrounding = " ".join(surround_parts)[:300]

        # Check for dialogue brackets
        dialogues = _RE_DIALOGUE.findall(para)
        dialogues_double = _RE_DIALOGUE_DOUBLE.findall(para)

        if dialogues or dialogues_double:
            for d in dialogues + dialogues_double:
                uid = f"ch{chapter.chapter_index:02d}_u{idx:04d}"
                utterances.append(Utterance(
                    utterance_id=uid,
                    chapter_index=chapter.chapter_index,
                    order_index=idx,
                    text=d,
                    utterance_type="dialogue",
                    surrounding_narration=surrounding,
                ))
                idx += 1
        elif _RE_THOUGHT.search(para):
            thoughts = _RE_THOUGHT.findall(para)
            for t in thoughts:
                uid = f"ch{chapter.chapter_index:02d}_u{idx:04d}"
                utterances.append(Utterance(
                    utterance_id=uid,
                    chapter_index=chapter.chapter_index,
                    order_index=idx,
                    text=t,
                    utterance_type="thought",
                    surrounding_narration=surrounding,
                ))
                idx += 1
        else:
            uid = f"ch{chapter.chapter_index:02d}_u{idx:04d}"
            utterances.append(Utterance(
                utterance_id=uid,
                chapter_index=chapter.chapter_index,
                order_index=idx,
                text=para,
                utterance_type="narration",
                surrounding_narration="",
            ))
            idx += 1

    # Extract mentioned character names from narration
    for u in utterances:
        if u.utterance_type == "narration":
            names = _RE_SPEAKER_VERB.findall(u.text)
            chapter.characters_mentioned.extend(names)

    # Deduplicate character mentions
    chapter.characters_mentioned = list(dict.fromkeys(chapter.characters_mentioned))
    chapter.utterances = utterances
    return utterances


# ── Phase C: LLM speaker inference ──────────────────────────────────────────

_LLM_SYSTEM_PROMPT = """あなたは日本語小説の話者推定アシスタントです。
与えられた章のテキストと発話一覧から、各発話の話者を推定してください。

ルール:
1. 地の文はnarrator（語り手）として扱う
2. 台詞（「」内）は文脈から話者を推定する
3. 心内語（（）内）は思考の主体を推定する
4. 直前の地の文に話者の手がかりがある場合はそれを優先する
5. 会話のターン交替パターン（AとBが交互に話す）を活用する
6. 確信度が低い場合は正直に低い値を返す
7. 話者不明の場合はspeakerをnullにし、候補を挙げる

JSON配列で回答してください。各要素:
{
  "utterance_id": "ch00_u0001",
  "speaker": "話者名" or null,
  "speaker_label": "未確定話者A" (speakerがnullの場合のラベル),
  "confidence": 0.0-1.0,
  "reason": "推定理由（日本語で簡潔に）",
  "candidates": [{"name": "候補名", "score": 0.0-1.0}]
}"""

_BATCH_SIZE = 15  # utterances per LLM call


def infer_speakers_with_llm(
    chapter: ChapterContext,
    known_characters: Optional[List[str]] = None,
    progress_callback: Optional[Callable] = None,
) -> List[Utterance]:
    """Infer speakers for all utterances in a chapter using LLM."""
    api_url = os.environ.get("LLM_API_URL", "")
    if not api_url:
        logger.warning("[speaker_inference] LLM_API_URL not set, returning utterances without speaker inference")
        return chapter.utterances

    api_key = os.environ.get("LLM_API_KEY", "")
    model = os.environ.get("LLM_MODEL", "")

    # Only process dialogue and thought utterances
    dialogue_utterances = [u for u in chapter.utterances if u.utterance_type in ("dialogue", "thought")]
    narration_utterances = [u for u in chapter.utterances if u.utterance_type == "narration"]

    # Set narrator for narration
    for u in narration_utterances:
        u.speaker = "narrator"
        u.confidence = 0.95
        u.reason = "地の文"

    if not dialogue_utterances:
        return chapter.utterances

    # Build context for LLM
    chars = known_characters or chapter.characters_mentioned or []
    chapter_summary = chapter.text[:1000]  # First 1000 chars as context

    # Process in batches
    for batch_start in range(0, len(dialogue_utterances), _BATCH_SIZE):
        batch = dialogue_utterances[batch_start:batch_start + _BATCH_SIZE]

        utterance_data = []
        for u in batch:
            utterance_data.append({
                "utterance_id": u.utterance_id,
                "text": u.text[:200],  # Truncate long text
                "type": u.utterance_type,
                "surrounding": u.surrounding_narration[:200],
            })

        user_prompt = _build_inference_prompt(chapter_summary, chars, utterance_data)

        try:
            results = _call_llm_inference(api_url, api_key, model, user_prompt)

            # Apply results to utterances
            result_map = {r.get("utterance_id"): r for r in results}
            for u in batch:
                if u.utterance_id in result_map:
                    r = result_map[u.utterance_id]
                    u.speaker = r.get("speaker")
                    u.speaker_label = r.get("speaker_label", "")
                    u.confidence = float(r.get("confidence", 0.0))
                    u.reason = r.get("reason", "")
                    u.candidates = r.get("candidates", [])
                    u.needs_review = u.confidence < 0.55

            # Touch LLM last-used
            try:
                from app.orchestrator.services.llm_manager import touch_last_used
                touch_last_used()
            except ImportError:
                pass

        except Exception as e:
            logger.error(f"[speaker_inference] LLM batch failed: {e}")
            for u in batch:
                u.speaker = None
                u.speaker_label = "推定失敗"
                u.confidence = 0.0
                u.reason = f"LLM error: {str(e)[:100]}"
                u.needs_review = True

        if progress_callback:
            done = min(batch_start + _BATCH_SIZE, len(dialogue_utterances))
            progress_callback(done, len(dialogue_utterances))

    return chapter.utterances


def _build_inference_prompt(
    chapter_summary: str,
    characters: List[str],
    utterances: List[dict],
) -> str:
    parts = [
        "## 章の冒頭テキスト（文脈）",
        chapter_summary,
        "",
    ]
    if characters:
        parts.append(f"## 登場人物候補: {', '.join(characters)}")
        parts.append("")

    parts.append("## 推定対象の発話一覧")
    for u in utterances:
        parts.append(f"- [{u['utterance_id']}] ({u['type']}) 「{u['text']}」")
        if u.get("surrounding"):
            parts.append(f"  周辺の地の文: {u['surrounding']}")

    parts.append("")
    parts.append("上記の各発話について、話者をJSON配列で回答してください。")
    return "\n".join(parts)


def _call_llm_inference(
    api_url: str,
    api_key: str,
    model: str,
    user_prompt: str,
) -> List[dict]:
    """Call LLM API and parse speaker inference results."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "messages": [
            {"role": "system", "content": _LLM_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 2000,
    }
    if model:
        payload["model"] = model

    url = api_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = f"{url}/chat/completions"

    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

    # Strip markdown code fences
    content = re.sub(r"^```(?:json)?\s*", "", content.strip())
    content = re.sub(r"\s*```$", "", content.strip())

    parsed = json.loads(content)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict) and "results" in parsed:
        return parsed["results"]
    return []


# ── Phase D: Reconciliation ─────────────────────────────────────────────────

def reconcile_speaker_labels(chapter: ChapterContext) -> List[Utterance]:
    """Apply consistency checks within a chapter's utterances.

    Rules:
    1. Consecutive dialogue with no narration break → likely alternating speakers
    2. If speaker jumps unexpectedly, flag for review
    3. Propagate high-confidence speakers to adjacent low-confidence utterances
    """
    utterances = chapter.utterances
    dialogues = [u for u in utterances if u.utterance_type in ("dialogue", "thought")]

    if len(dialogues) < 2:
        return utterances

    # Rule 1: Turn alternation
    for i in range(1, len(dialogues)):
        prev = dialogues[i - 1]
        curr = dialogues[i]

        # If both have speakers and they're the same but adjacent, check for alternation
        if (prev.speaker and curr.speaker and prev.speaker == curr.speaker
                and prev.confidence > 0.7 and curr.confidence < 0.5):
            # Possible alternation error - flag for review
            curr.needs_review = True
            if not curr.reason:
                curr.reason = "連続同一話者: ターン交替の可能性あり"

    # Rule 2: Propagate high-confidence to adjacent low-confidence
    for i in range(len(dialogues)):
        curr = dialogues[i]
        if curr.confidence >= 0.5 or curr.speaker:
            continue

        # Look at neighbors
        prev_speaker = dialogues[i - 1].speaker if i > 0 else None
        next_speaker = dialogues[i + 1].speaker if i + 1 < len(dialogues) else None
        prev_conf = dialogues[i - 1].confidence if i > 0 else 0
        next_conf = dialogues[i + 1].confidence if i + 1 < len(dialogues) else 0

        # If both neighbors have the same speaker with high confidence, propagate
        if prev_speaker and prev_speaker == next_speaker and min(prev_conf, next_conf) > 0.7:
            curr.speaker = prev_speaker
            curr.confidence = min(prev_conf, next_conf) * 0.6
            curr.reason = f"前後の話者({prev_speaker})から伝搬"

    # Rule 3: Label unknown speakers with consistent labels
    _label_unknown_speakers(dialogues)

    return utterances


def _label_unknown_speakers(utterances: List[Utterance]) -> None:
    """Assign consistent labels to unknown speakers within a sequence."""
    unknown_counter = 0
    # Track unknown speakers by their candidate pattern
    unknown_labels: Dict[str, str] = {}

    for u in utterances:
        if u.speaker is None and u.utterance_type in ("dialogue", "thought"):
            # Create a key from top candidates
            candidate_key = ""
            if u.candidates:
                top = sorted(u.candidates, key=lambda c: c.get("score", 0), reverse=True)[:2]
                candidate_key = "|".join(c.get("name", "") for c in top)

            if candidate_key and candidate_key in unknown_labels:
                u.speaker_label = unknown_labels[candidate_key]
            else:
                unknown_counter += 1
                label = f"未確定話者{_to_letter(unknown_counter)}"
                u.speaker_label = label
                if candidate_key:
                    unknown_labels[candidate_key] = label

            u.needs_review = True


def _to_letter(n: int) -> str:
    """Convert 1→A, 2→B, etc."""
    return chr(64 + min(n, 26))


# ── Export to AnnotatedSegment format ────────────────────────────────────────

def export_speaker_segments(
    chapters: List[ChapterContext],
) -> List[dict]:
    """Convert chapter utterances to AnnotatedSegment-compatible dicts."""
    segments = []
    global_order = 0

    for chapter in chapters:
        for u in chapter.utterances:
            seg = {
                "chapter_index": chapter.chapter_index,
                "order_index": global_order,
                "raw_text": u.text,
                "normalized_text": u.text,
                "segment_type": u.utterance_type,
                "predicted_speaker": u.speaker or u.speaker_label or "unknown",
                "confidence": u.confidence,
                "reason": u.reason,
                "is_chapter_header": False,
                "candidates": u.candidates,
                "needs_review": u.needs_review,
                "evidence_spans": [],
            }
            segments.append(seg)
            global_order += 1

    return segments


# ── Main pipeline function ───────────────────────────────────────────────────

def run_llm_speaker_pipeline(
    text: str,
    known_characters: Optional[List[str]] = None,
    progress_callback: Optional[Callable] = None,
) -> List[dict]:
    """Run the full LLM-primary speaker inference pipeline.

    Args:
        text: Full novel text (OCR-cleaned).
        known_characters: Optional list of known character names.
        progress_callback: Optional (stage, detail) callback.

    Returns:
        List of AnnotatedSegment-compatible dicts.
    """
    from app.orchestrator.services.llm_manager import acquire, release

    if progress_callback:
        progress_callback("splitting", "章分割中...")

    # Phase A: Split into chapters
    chapters = split_into_chapters(text)
    logger.info(f"[speaker_inference] Split into {len(chapters)} chapters")

    if progress_callback:
        progress_callback("extracting", f"{len(chapters)}章から発話抽出中...")

    # Phase B: Extract utterances per chapter
    total_utterances = 0
    for ch in chapters:
        extract_utterances(ch)
        total_utterances += len(ch.utterances)

    logger.info(f"[speaker_inference] Extracted {total_utterances} utterances total")

    # Phase C: LLM speaker inference
    acquire("speaker_split")
    try:
        for i, ch in enumerate(chapters):
            if progress_callback:
                progress_callback("inferring", f"章{i+1}/{len(chapters)} 話者推定中...")

            dialogue_count = sum(1 for u in ch.utterances if u.utterance_type in ("dialogue", "thought"))
            if dialogue_count > 0:
                infer_speakers_with_llm(ch, known_characters)

            # Phase D: Reconcile within chapter
            reconcile_speaker_labels(ch)

    finally:
        release("speaker_split")

    if progress_callback:
        progress_callback("exporting", "結果出力中...")

    # Export
    segments = export_speaker_segments(chapters)
    logger.info(f"[speaker_inference] Pipeline complete: {len(segments)} segments")
    return segments
