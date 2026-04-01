"""Speaker segmentation: rule-based pre-pass + LLM estimation."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional

import requests

from app.orchestrator.services.preprocessor import RawSegment
from app.shared.logger import get_logger

logger = get_logger("speaker_segmenter")

# ── Data ──────────────────────────────────────────────────────────────────────

@dataclass
class AnnotatedSegment:
    chapter_index: int
    order_index: int
    raw_text: str
    normalized_text: str
    segment_type: str  # narration / dialogue / thought / unknown
    predicted_speaker: str
    confidence: float
    reason: str
    is_chapter_header: bool = False


# ── Rule-based pre-pass ───────────────────────────────────────────────────────

_DIALOGUE_RE = re.compile(r"^[「『](.+)[」』]$", re.DOTALL)
_THOUGHT_RE = re.compile(r"^[（(](.+)[）)]$", re.DOTALL)
_NARR_CLUES = re.compile(r"(と言った|と答えた|と叫んだ|と呟いた|と続けた|と笑った|と怒った|と泣いた|は言った|は答えた|が言った)", re.IGNORECASE)


def _rule_based_classify(text: str) -> tuple[str, float]:
    """Return (segment_type, confidence)."""
    t = text.strip()
    if _DIALOGUE_RE.match(t):
        return "dialogue", 0.85
    if _THOUGHT_RE.match(t):
        return "thought", 0.80
    # Short text with dialogue marker nearby
    if "「" in t or "『" in t:
        return "dialogue", 0.60
    if _NARR_CLUES.search(t):
        return "narration", 0.70
    return "narration", 0.50


def rule_based_pass(segments: List[RawSegment]) -> List[AnnotatedSegment]:
    annotated: List[AnnotatedSegment] = []
    for seg in segments:
        text = seg.text.strip()
        if seg.is_chapter_header:
            annotated.append(AnnotatedSegment(
                chapter_index=seg.chapter_index,
                order_index=seg.order_index,
                raw_text=text,
                normalized_text=text,
                segment_type="narration",
                predicted_speaker="narrator",
                confidence=0.95,
                reason="chapter header",
                is_chapter_header=True,
            ))
            continue
        stype, conf = _rule_based_classify(text)
        speaker = "narrator" if stype == "narration" else "unknown"
        annotated.append(AnnotatedSegment(
            chapter_index=seg.chapter_index,
            order_index=seg.order_index,
            raw_text=text,
            normalized_text=text,
            segment_type=stype,
            predicted_speaker=speaker,
            confidence=conf,
            reason="rule-based",
        ))
    return annotated


# ── LLM adapter ───────────────────────────────────────────────────────────────

LLM_API_URL = os.environ.get("LLM_API_URL", "")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL   = os.environ.get("LLM_MODEL", "gpt-4o-mini")
LLM_CONTEXT_WINDOW = 8  # segments around target for context

_SYSTEM_PROMPT = """あなたは日本語の小説・ラノベの話者分割AIです。
与えられた各セグメントに対して、以下のJSON配列を返してください。
各要素は {"order_index": int, "segment_type": "narration"|"dialogue"|"thought"|"unknown",
          "predicted_speaker": string, "confidence": float(0-1), "reason": string}

注意:
- dialogueは「」『』に囲まれた発話
- thoughtは（）に囲まれた心内文
- narrationは地の文
- predicted_speakerは「narrator」「主人公」「ヒロイン」など推定できる名前、不明は「unknown」
- confidenceは確信度0.0〜1.0
- reasonは日本語で簡潔に

JSONのみ返してください。説明文は不要です。"""


def _llm_annotate_batch(
    segments: List[AnnotatedSegment], batch: List[AnnotatedSegment]
) -> List[dict]:
    """Call LLM for a batch and return list of dicts."""
    if not LLM_API_URL:
        return []

    # build context
    all_indices = {s.order_index: s for s in segments}
    target_idx = [s.order_index for s in batch]

    # include surrounding context
    context_set = set(target_idx)
    for idx in target_idx:
        for d in range(-LLM_CONTEXT_WINDOW, LLM_CONTEXT_WINDOW + 1):
            context_set.add(idx + d)

    context_segs = sorted(
        [s for s in segments if s.order_index in context_set],
        key=lambda s: s.order_index,
    )

    user_msg_parts = []
    for s in context_segs:
        mark = " [TARGET]" if s.order_index in target_idx else ""
        user_msg_parts.append(f"[{s.order_index}]{mark}: {s.normalized_text[:120]}")

    user_content = "\n".join(user_msg_parts)
    user_content += f"\n\nTARGETのorder_indexは {target_idx} です。これらのみ結果に含めてください。"

    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.2,
        "max_tokens": 1500,
    }

    try:
        resp = requests.post(
            f"{LLM_API_URL.rstrip('/')}/chat/completions",
            json=payload,
            headers=headers,
            timeout=60,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        # strip markdown code fences if present
        content = re.sub(r"```json\s*", "", content)
        content = re.sub(r"```\s*", "", content)
        return json.loads(content)
    except Exception as e:
        logger.warning(f"LLM call failed: {e}")
        return []


# ── Main entry point ──────────────────────────────────────────────────────────

BATCH_SIZE = 20  # segments per LLM call


def segment_speakers(raw_segments: List[RawSegment]) -> List[AnnotatedSegment]:
    """Full pipeline: rule-based → LLM refinement."""
    annotated = rule_based_pass(raw_segments)

    if not LLM_API_URL:
        logger.info("LLM_API_URL not set – using rule-based only")
        _apply_speaker_propagation(annotated)
        return annotated

    # Batch LLM calls for non-chapter, low-confidence segments
    targets = [
        s for s in annotated
        if not s.is_chapter_header and (s.confidence < 0.8 or s.predicted_speaker == "unknown")
    ]
    logger.info(f"LLM refinement for {len(targets)} segments")

    # group into batches
    for i in range(0, len(targets), BATCH_SIZE):
        batch = targets[i : i + BATCH_SIZE]
        results = _llm_annotate_batch(annotated, batch)
        result_map = {r["order_index"]: r for r in results}

        for seg in batch:
            if seg.order_index in result_map:
                r = result_map[seg.order_index]
                seg.segment_type = r.get("segment_type", seg.segment_type)
                seg.predicted_speaker = r.get("predicted_speaker", seg.predicted_speaker)
                seg.confidence = float(r.get("confidence", seg.confidence))
                seg.reason = r.get("reason", seg.reason) + " (llm)"

    _apply_speaker_propagation(annotated)
    return annotated


def _apply_speaker_propagation(segments: List[AnnotatedSegment]) -> None:
    """Propagate known speakers to adjacent unknown dialogue segments."""
    last_known: Optional[str] = None
    for seg in segments:
        if seg.predicted_speaker not in ("unknown", "narrator", ""):
            last_known = seg.predicted_speaker
        elif seg.segment_type == "dialogue" and last_known and seg.predicted_speaker == "unknown":
            seg.predicted_speaker = last_known
            seg.confidence = min(seg.confidence, 0.45)
            seg.reason += " (propagated)"


# ── Speaker consolidation ─────────────────────────────────────────────────────

_ALIAS_GROUPS = [
    # add project-specific patterns here
]

def build_speaker_list(segments: List[AnnotatedSegment]) -> List[dict]:
    """Return unique speakers with counts."""
    from collections import Counter
    counts: Counter = Counter()
    for seg in segments:
        sp = seg.predicted_speaker or "unknown"
        counts[sp] += 1

    speakers = []
    for name, count in counts.most_common():
        speakers.append({"name": name, "segment_count": count, "aliases": []})
    return speakers
