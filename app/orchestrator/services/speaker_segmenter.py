"""Speaker segmentation: rule-based pre-pass + LLM estimation.

Supports a configurable DiarizationConfig so users can pick/combine rules
interactively (diarization studio). Falls back to the default pipeline when
no config is supplied (backwards compatible).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
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


# ── Rule catalogue ─────────────────────────────────────────────────────────────

AVAILABLE_RULES: dict[str, dict] = {
    "dialogue_brackets": {
        "label": "対話括弧ルール「」『』",
        "description": "「」『』で囲まれたセグメントを dialogue として分類します",
        "params": {},
    },
    "thought_brackets": {
        "label": "心内文括弧ルール（）",
        "description": "（）で囲まれたセグメントを thought として分類します",
        "params": {},
    },
    "narration_clues": {
        "label": "語り手キーワードルール",
        "description": "「と言った」などの語りキーワードを含むセグメントを narration として分類します",
        "params": {
            "keywords": {
                "type": "list",
                "default": [
                    "と言った", "と答えた", "と叫んだ", "と呟いた",
                    "と続けた", "と笑った", "と怒った", "と泣いた",
                    "は言った", "は答えた", "が言った",
                ],
            },
        },
    },
    "speaker_propagation": {
        "label": "話者伝搬ルール",
        "description": "直前の既知話者を隣接する unknown dialogue に伝搬します",
        "params": {},
    },
    "llm_refinement": {
        "label": "LLM推定ルール",
        "description": "llama.cpp サーバーを使って低信頼度セグメントを再推定します",
        "params": {
            "confidence_threshold": {
                "type": "float",
                "default": 0.8,
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
            },
        },
    },
}

DEFAULT_RULE_ORDER = list(AVAILABLE_RULES.keys())


@dataclass
class RuleSpec:
    rule_id: str
    enabled: bool = True
    params: dict = field(default_factory=dict)


@dataclass
class DiarizationConfig:
    rules: List[RuleSpec] = field(
        default_factory=lambda: [RuleSpec(r) for r in DEFAULT_RULE_ORDER]
    )
    # LLM fallback mode: "disabled" | "on_error" | "always"
    # - disabled: no LLM usage
    # - on_error: LLM used only for low-confidence / unknown speakers (default behaviour)
    # - always: LLM normalises / cleans every segment text before saving
    llm_fallback_mode: str = "on_error"

    def get_rule(self, rule_id: str) -> Optional[RuleSpec]:
        for r in self.rules:
            if r.rule_id == rule_id and r.enabled:
                return r
        return None


@dataclass
class LlmBatchError:
    """Captures failure details for a single LLM annotation batch."""
    batch_order_indices: List[int]
    exception_type: str
    message: str
    raw_response_excerpt: str  # first 300 chars of LLM response (if available)


# ── Rule-based pre-pass ───────────────────────────────────────────────────────

_DIALOGUE_RE = re.compile(r"^[「『](.+)[」』]$", re.DOTALL)
_THOUGHT_RE = re.compile(r"^[（(](.+)[）)]$", re.DOTALL)
_DEFAULT_NARR_CLUES = re.compile(
    r"(と言った|と答えた|と叫んだ|と呟いた|と続けた|と笑った|と怒った|と泣いた|は言った|は答えた|が言った)",
    re.IGNORECASE,
)


def _build_narr_re(keywords: list[str]) -> re.Pattern:
    escaped = [re.escape(k) for k in keywords]
    return re.compile("(" + "|".join(escaped) + ")", re.IGNORECASE)


def _rule_based_classify(
    text: str,
    *,
    use_dialogue: bool = True,
    use_thought: bool = True,
    narr_re: Optional[re.Pattern] = None,
) -> tuple[str, float]:
    """Return (segment_type, confidence)."""
    t = text.strip()
    if use_dialogue and _DIALOGUE_RE.match(t):
        return "dialogue", 0.85
    if use_thought and _THOUGHT_RE.match(t):
        return "thought", 0.80
    if use_dialogue and ("「" in t or "『" in t):
        return "dialogue", 0.60
    clues = narr_re if narr_re is not None else _DEFAULT_NARR_CLUES
    if clues and clues.search(t):
        return "narration", 0.70
    return "narration", 0.50


def rule_based_pass(
    segments: List[RawSegment],
    config: Optional[DiarizationConfig] = None,
) -> List[AnnotatedSegment]:
    use_dialogue = True
    use_thought = True
    narr_re: Optional[re.Pattern] = None

    if config is not None:
        use_dialogue = config.get_rule("dialogue_brackets") is not None
        use_thought = config.get_rule("thought_brackets") is not None
        narr_spec = config.get_rule("narration_clues")
        if narr_spec is not None:
            kws = narr_spec.params.get("keywords", AVAILABLE_RULES["narration_clues"]["params"]["keywords"]["default"])
            narr_re = _build_narr_re(kws)
        else:
            narr_re = re.compile(r"(?!)")  # never matches

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
        stype, conf = _rule_based_classify(
            text,
            use_dialogue=use_dialogue,
            use_thought=use_thought,
            narr_re=narr_re,
        )
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


def _get_llm_api_url() -> str:
    """Dynamically read LLM_API_URL so llm_manager changes take effect."""
    return os.environ.get("LLM_API_URL", "")


def _get_llm_api_key() -> str:
    return os.environ.get("LLM_API_KEY", "")


def _get_llm_model() -> str:
    return os.environ.get("LLM_MODEL", "gpt-4o-mini")


def _llm_annotate_batch(
    segments: List[AnnotatedSegment],
    batch: List[AnnotatedSegment],
) -> tuple[List[dict], Optional[LlmBatchError]]:
    """Call LLM for a batch.

    Returns (results, error_info).
    - On success: (list_of_dicts, None)
    - On failure: ([], LlmBatchError)
    """
    llm_url = _get_llm_api_url()
    if not llm_url:
        return [], None

    target_idx = [s.order_index for s in batch]

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
    api_key = _get_llm_api_key()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": _get_llm_model(),
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.2,
        "max_tokens": 1500,
    }

    raw_content = ""
    try:
        resp = requests.post(
            f"{llm_url.rstrip('/')}/chat/completions",
            json=payload,
            headers=headers,
            timeout=60,
        )
        resp.raise_for_status()
        # Touch last_used_at so auto-unload timer resets on LLM activity
        try:
            from app.orchestrator.services.llm_manager import touch_last_used
            touch_last_used()
        except Exception:
            pass
        raw_content = resp.json()["choices"][0]["message"]["content"].strip()
        cleaned = re.sub(r"```json\s*", "", raw_content)
        cleaned = re.sub(r"```\s*", "", cleaned)
        return json.loads(cleaned), None
    except json.JSONDecodeError as e:
        err = LlmBatchError(
            batch_order_indices=target_idx,
            exception_type="JSONDecodeError",
            message=str(e),
            raw_response_excerpt=raw_content[:300],
        )
        logger.warning(f"LLM JSON parse failed for batch {target_idx}: {e} | response excerpt: {raw_content[:200]!r}")
        return [], err
    except Exception as e:
        err = LlmBatchError(
            batch_order_indices=target_idx,
            exception_type=type(e).__name__,
            message=str(e),
            raw_response_excerpt=raw_content[:300] if raw_content else "",
        )
        logger.warning(f"LLM call failed for batch {target_idx}: {e}")
        return [], err


# ── Main entry point ──────────────────────────────────────────────────────────

BATCH_SIZE = 20


def segment_speakers(
    raw_segments: List[RawSegment],
    config: Optional[DiarizationConfig] = None,
    progress_cb: Optional[object] = None,  # callable(stage, pct, current, total) or None
) -> tuple[List[AnnotatedSegment], List[dict]]:
    """Full pipeline: rule-based → LLM refinement.

    Returns (annotated_segments, llm_errors).
    ``llm_errors`` is a list of error detail dicts (empty on success).

    ``config`` is optional. When omitted the full default pipeline runs,
    which is equivalent to all rules enabled – matching original behaviour.

    ``progress_cb`` is an optional callback for reporting progress to the job tracker.
    """
    import time as _time

    def _cb(stage: str, pct: int, cur: int = 0, total: int = 0) -> None:
        if progress_cb is not None:
            try:
                progress_cb(stage, pct, cur, total)
            except Exception:
                pass

    total_segs = len(raw_segments)
    _cb("rule_based", 10, 0, total_segs)

    annotated = rule_based_pass(raw_segments, config=config)
    llm_errors: List[dict] = []

    llm_url = _get_llm_api_url()
    run_llm = llm_url != ""
    if config is not None:
        run_llm = run_llm and (config.get_rule("llm_refinement") is not None)

    if not run_llm:
        logger.info("LLM refinement disabled – using rule-based only")
        _cb("speaker_propagation", 85, 0, total_segs)
        _maybe_apply_propagation(annotated, config)
        _cb("complete", 100, total_segs, total_segs)
        return annotated, llm_errors

    llm_spec = config.get_rule("llm_refinement") if config else None
    conf_threshold = float(
        (llm_spec.params.get("confidence_threshold") if llm_spec else None) or 0.8
    )

    targets = [
        s for s in annotated
        if not s.is_chapter_header and (s.confidence < conf_threshold or s.predicted_speaker == "unknown")
    ]
    total_batches = max(1, (len(targets) + BATCH_SIZE - 1) // BATCH_SIZE)
    logger.info(
        f"LLM refinement for {len(targets)} segments in {total_batches} batches "
        f"(threshold={conf_threshold})"
    )

    for i in range(0, len(targets), BATCH_SIZE):
        batch_num = i // BATCH_SIZE + 1
        batch = targets[i : i + BATCH_SIZE]

        # Progress: 20-80% range for LLM batches
        batch_pct = 20 + int((batch_num / total_batches) * 60)
        _cb("llm_batch", batch_pct, batch_num, total_batches)

        batch_start = _time.time()
        results, err = _llm_annotate_batch(annotated, batch)
        batch_elapsed = round(_time.time() - batch_start, 1)

        logger.info(
            f"LLM batch {batch_num}/{total_batches}: "
            f"{len(batch)} segments, elapsed={batch_elapsed}s, "
            f"{'OK' if err is None else 'FAILED'}"
        )

        if err is not None:
            context_snippets = []
            for seg in batch[:3]:
                context_snippets.append({
                    "order_index": seg.order_index,
                    "text_excerpt": seg.normalized_text[:80],
                    "predicted_speaker": seg.predicted_speaker,
                    "confidence": round(seg.confidence, 3),
                })
            llm_errors.append({
                "batch_order_indices": err.batch_order_indices,
                "exception_type": err.exception_type,
                "message": err.message,
                "raw_response_excerpt": err.raw_response_excerpt,
                "context_snippets": context_snippets,
            })
            logger.warning(
                f"LLM batch failed: indices={err.batch_order_indices} "
                f"type={err.exception_type} msg={err.message}"
            )
            continue

        result_map = {r["order_index"]: r for r in results}

        for seg in batch:
            if seg.order_index in result_map:
                r = result_map[seg.order_index]
                seg.segment_type = r.get("segment_type", seg.segment_type)
                seg.predicted_speaker = r.get("predicted_speaker", seg.predicted_speaker)
                seg.confidence = float(r.get("confidence", seg.confidence))
                seg.reason = r.get("reason", seg.reason) + " (llm)"

    _cb("speaker_propagation", 85, 0, total_segs)
    _maybe_apply_propagation(annotated, config)
    return annotated, llm_errors


def _maybe_apply_propagation(
    segments: List[AnnotatedSegment],
    config: Optional[DiarizationConfig],
) -> None:
    if config is None or config.get_rule("speaker_propagation") is not None:
        _apply_speaker_propagation(segments)


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
