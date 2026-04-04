"""Speaker segmentation: rule-based pre-pass + LLM estimation.

Supports a configurable DiarizationConfig so users can pick/combine rules
interactively (diarization studio). Falls back to the default pipeline when
no config is supplied (backwards compatible).

v2 enhanced pipeline adds:
  - monologue_detection rule
  - candidate generation per segment
  - constrained JSON LLM prompt with retry
  - global consistency 2nd pass
  - AnnotatedSegment extended with candidates/evidence_spans/needs_review/rule_log
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

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
    segment_type: str  # narration / dialogue / thought / monologue / unknown
    predicted_speaker: str
    confidence: float
    reason: str
    is_chapter_header: bool = False
    # Enhanced pipeline fields (all have defaults → backward compatible)
    monologue_subtype: Optional[str] = None              # "inner" | None
    candidates: List[dict] = field(default_factory=list) # CandidateSpeaker dicts
    evidence_spans: List[str] = field(default_factory=list)
    needs_review: bool = False
    rule_log: List[dict] = field(default_factory=list)   # [{"rule": str, "fired": bool}]
    character_id: Optional[str] = None                   # FK to character_master.id


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
    "monologue_detection": {
        "label": "独白・心内語検出",
        "description": "「〜と思った」などのパターンで独白(monologue)を検出します",
        "params": {},
    },
    "llm_primary": {
        "label": "LLM主体話者推定",
        "description": "LLMを主体とした章単位の話者推定を行います（高精度、低速）",
        "params": {},
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


_MONOLOGUE_DETECT_RE = re.compile(
    r"(〜?と思った|〜?と考えた|〜?と感じた|心の中で|心中で|胸の内で"
    r"|と心の中で|心の奥で|頭の中で|脳裏に|胸の中で"
    r"|と思う|と考える|と感じる|と悟った|と気づいた)"
)


def _rule_based_classify(
    text: str,
    *,
    use_dialogue: bool = True,
    use_thought: bool = True,
    use_monologue: bool = False,
    narr_re: Optional[re.Pattern] = None,
) -> tuple[str, float, Optional[str], List[dict]]:
    """Return (segment_type, confidence, monologue_subtype, rule_log)."""
    t = text.strip()
    rule_log: List[dict] = []

    # Dialogue bracket (full match)
    if use_dialogue and _DIALOGUE_RE.match(t):
        rule_log.append({"rule": "dialogue_brackets_full", "fired": True})
        return "dialogue", 0.85, None, rule_log

    # Thought bracket
    if use_thought and _THOUGHT_RE.match(t):
        rule_log.append({"rule": "thought_brackets", "fired": True})
        return "thought", 0.80, None, rule_log

    # Monologue detection (inner thought without brackets)
    if use_monologue and _MONOLOGUE_DETECT_RE.search(t):
        rule_log.append({"rule": "monologue_detection", "fired": True})
        return "monologue", 0.72, "inner", rule_log

    # Dialogue bracket (partial / nested)
    if use_dialogue and ("「" in t or "『" in t):
        rule_log.append({"rule": "dialogue_brackets_partial", "fired": True})
        return "dialogue", 0.60, None, rule_log

    # Narration clues
    clues = narr_re if narr_re is not None else _DEFAULT_NARR_CLUES
    if clues and clues.search(t):
        rule_log.append({"rule": "narration_clues", "fired": True})
        return "narration", 0.70, None, rule_log

    rule_log.append({"rule": "default_narration", "fired": True})
    return "narration", 0.50, None, rule_log


def rule_based_pass(
    segments: List[RawSegment],
    config: Optional[DiarizationConfig] = None,
) -> List[AnnotatedSegment]:
    use_dialogue = True
    use_thought = True
    use_monologue = False
    narr_re: Optional[re.Pattern] = None

    if config is not None:
        use_dialogue = config.get_rule("dialogue_brackets") is not None
        use_thought = config.get_rule("thought_brackets") is not None
        use_monologue = config.get_rule("monologue_detection") is not None
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
                rule_log=[{"rule": "chapter_header", "fired": True}],
            ))
            continue
        stype, conf, monologue_subtype, rule_log = _rule_based_classify(
            text,
            use_dialogue=use_dialogue,
            use_thought=use_thought,
            use_monologue=use_monologue,
            narr_re=narr_re,
        )
        speaker = "narrator" if stype == "narration" else "unknown"
        # Monologue is typically the viewpoint character
        if stype == "monologue":
            speaker = "protagonist"
        annotated.append(AnnotatedSegment(
            chapter_index=seg.chapter_index,
            order_index=seg.order_index,
            raw_text=text,
            normalized_text=text,
            segment_type=stype,
            predicted_speaker=speaker,
            confidence=conf,
            reason="rule-based",
            monologue_subtype=monologue_subtype,
            rule_log=rule_log,
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


# ── Enhanced pipeline (v2) ────────────────────────────────────────────────────

_SYSTEM_PROMPT_V2 = """あなたは日本語ラノベの話者識別AIです。
各セグメントについてJSON配列を返してください。

出力形式（必ずこの形式のみ）:
[
  {
    "order_index": <int>,
    "speaker": "<candidatesのcharacter_idまたは'unknown'>",
    "confidence": <float 0.0-1.0>,
    "evidence_spans": ["<根拠テキスト30字以内>"],
    "reason_short": "<30字以内の日本語>",
    "needs_review": <true/false>
  }
]

制約:
- speaker は必ず渡された candidates リストの character_id か "unknown" のみ使用すること
- needs_review=true にする条件: confidence < 0.55 または上位2候補のスコア差 < 0.1
- evidence_spans は最大2件、各30字以内のテキスト抜粋
- JSONのみ返すこと。説明文は一切不要。"""


def _validate_llm_v2_item(item: dict, valid_ids: set) -> bool:
    """Validate a single v2 LLM response item."""
    required = {"order_index", "speaker", "confidence", "evidence_spans",
                "reason_short", "needs_review"}
    if not required.issubset(item.keys()):
        return False
    if item["speaker"] not in valid_ids and item["speaker"] != "unknown":
        return False
    if not (0.0 <= float(item.get("confidence", -1)) <= 1.0):
        return False
    return True


def _format_candidates_for_prompt(candidate_dicts: List[dict]) -> str:
    """Format candidate list as a compact string for the LLM prompt."""
    parts = []
    for c in candidate_dicts:
        ev = "; ".join(c.get("evidence", [])[:2])
        parts.append(f"  - id={c['character_id']} name={c['name']} "
                     f"prior={c['confidence']:.2f} ev=[{ev}]")
    return "\n".join(parts) if parts else "  (候補なし)"


def _llm_annotate_batch_v2(
    segments: List[AnnotatedSegment],
    batch: List[AnnotatedSegment],
    candidate_map: Dict[int, List[dict]],  # order_index → candidate dicts
) -> tuple[List[dict], Optional[LlmBatchError]]:
    """Call LLM with candidate-constrained prompt (v2).

    Retries up to 2 extra times on JSON parse or validation failure.
    Returns (results, error_info).
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
        if s.order_index in target_idx:
            cands = candidate_map.get(s.order_index, [])
            cand_str = _format_candidates_for_prompt(cands)
            valid_ids = {c["character_id"] for c in cands} | {"unknown"}
            user_msg_parts.append(
                f"[{s.order_index}][TARGET] type={s.segment_type}: "
                f"{s.normalized_text[:150]}\n"
                f"candidates:\n{cand_str}"
            )
        else:
            user_msg_parts.append(
                f"[{s.order_index}] type={s.segment_type} "
                f"speaker={s.predicted_speaker}: {s.normalized_text[:80]}"
            )

    # Collect valid IDs for validation
    all_valid_ids: set = set()
    for idx in target_idx:
        cands = candidate_map.get(idx, [])
        all_valid_ids.update(c["character_id"] for c in cands)
    all_valid_ids.add("unknown")

    user_content = "\n---\n".join(user_msg_parts)
    user_content += f"\n\nTARGETのorder_indexは {target_idx} です。これらのみ出力してください。"

    headers = {"Content-Type": "application/json"}
    api_key = _get_llm_api_key()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": _get_llm_model(),
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT_V2},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.1,
        "max_tokens": 2000,
    }

    last_err: Optional[LlmBatchError] = None
    raw_content = ""
    for attempt in range(3):
        try:
            resp = requests.post(
                f"{llm_url.rstrip('/')}/chat/completions",
                json=payload,
                headers=headers,
                timeout=90,
            )
            resp.raise_for_status()
            try:
                from app.orchestrator.services.llm_manager import touch_last_used
                touch_last_used()
            except Exception:
                pass
            raw_content = resp.json()["choices"][0]["message"]["content"].strip()
            cleaned = re.sub(r"```json\s*", "", raw_content)
            cleaned = re.sub(r"```\s*", "", cleaned)
            parsed = json.loads(cleaned)
            # Validate each item
            valid = [item for item in parsed if _validate_llm_v2_item(item, all_valid_ids)]
            if valid:
                return valid, None
            # All items failed validation – retry
            logger.warning(
                f"LLM v2 attempt {attempt+1}: all items failed validation, "
                f"retrying... (response excerpt: {raw_content[:200]!r})"
            )
        except json.JSONDecodeError as e:
            logger.warning(f"LLM v2 attempt {attempt+1} JSON error: {e}")
            last_err = LlmBatchError(
                batch_order_indices=target_idx,
                exception_type="JSONDecodeError",
                message=str(e),
                raw_response_excerpt=raw_content[:300],
            )
        except Exception as e:
            logger.warning(f"LLM v2 attempt {attempt+1} error: {e}")
            last_err = LlmBatchError(
                batch_order_indices=target_idx,
                exception_type=type(e).__name__,
                message=str(e),
                raw_response_excerpt=raw_content[:300] if raw_content else "",
            )

    if last_err is None:
        last_err = LlmBatchError(
            batch_order_indices=target_idx,
            exception_type="ValidationError",
            message="All attempts produced invalid response",
            raw_response_excerpt=raw_content[:300],
        )
    return [], last_err


def segment_speakers_enhanced(
    raw_segments: List[RawSegment],
    config: Optional[DiarizationConfig] = None,
    char_dict=None,           # Optional[CharacterDict]
    project_id: Optional[str] = None,
    db=None,                  # Optional SQLAlchemy Session
    progress_cb=None,
) -> tuple[List[AnnotatedSegment], List[dict]]:
    """Enhanced v2 pipeline.

    Stages:
      1. rule_based_pass  (with monologue_detection enabled)
      2. build_character_dict  →  persist to DB
      3. generate_candidates  per dialogue/thought/unknown/monologue segment
      4. _llm_annotate_batch_v2  with candidate maps
      5. apply_global_consistency
      6. _apply_speaker_propagation

    The original segment_speakers() is NOT modified – callers opt in via
    use_enhanced_pipeline=True in the request body.
    """
    import time as _time
    from app.orchestrator.services.character_builder import (
        build_character_dict, persist_character_dict,
    )
    from app.orchestrator.services.candidate_generator import (
        generate_candidates, candidate_set_to_dicts,
    )
    from app.orchestrator.services.global_consistency import apply_global_consistency

    def _cb(stage: str, pct: int, cur: int = 0, total: int = 0) -> None:
        if progress_cb is not None:
            try:
                progress_cb(stage, pct, cur, total)
            except Exception:
                pass

    total_segs = len(raw_segments)
    llm_errors: List[dict] = []

    # ── Stage 1: rule-based pass with monologue detection ─────────────────────
    _cb("rule_based", 5, 0, total_segs)
    # Enable monologue detection in the config
    enhanced_config = _make_enhanced_config(config)
    annotated = rule_based_pass(raw_segments, config=enhanced_config)

    # ── Stage 2: build character dictionary ──────────────────────────────────
    _cb("char_dict_build", 15, 0, total_segs)
    if char_dict is None:
        existing_names = [s.predicted_speaker for s in annotated
                          if s.predicted_speaker not in ("unknown", "narrator", "protagonist", "")]
        char_dict = build_character_dict(annotated, existing_names)

    if db is not None and project_id is not None:
        try:
            persist_character_dict(char_dict, project_id, db)
        except Exception as e:
            logger.warning(f"persist_character_dict failed: {e}")

    # ── Stage 3: candidate generation ────────────────────────────────────────
    _cb("candidate_gen", 25, 0, total_segs)
    target_types = {"dialogue", "thought", "monologue", "unknown"}
    candidate_map: Dict[int, List[dict]] = {}

    for seg in annotated:
        if seg.is_chapter_header or seg.segment_type not in target_types:
            continue
        cs = generate_candidates(seg, annotated, char_dict)
        cand_dicts = candidate_set_to_dicts(cs)
        candidate_map[seg.order_index] = cand_dicts
        seg.candidates = cand_dicts

    # ── Stage 4: LLM re-ranking ───────────────────────────────────────────────
    llm_url = _get_llm_api_url()
    run_llm = bool(llm_url)
    if config is not None:
        run_llm = run_llm and (config.get_rule("llm_refinement") is not None)

    if run_llm:
        llm_spec = config.get_rule("llm_refinement") if config else None
        conf_threshold = float(
            (llm_spec.params.get("confidence_threshold") if llm_spec else None) or 0.8
        )
        targets = [
            s for s in annotated
            if not s.is_chapter_header
            and s.segment_type in target_types
            and (s.confidence < conf_threshold or s.predicted_speaker == "unknown")
        ]
        total_batches = max(1, (len(targets) + BATCH_SIZE - 1) // BATCH_SIZE)
        logger.info(
            f"LLM v2 refinement: {len(targets)} segments in {total_batches} batches"
        )

        for i in range(0, len(targets), BATCH_SIZE):
            batch_num = i // BATCH_SIZE + 1
            batch = targets[i: i + BATCH_SIZE]
            batch_pct = 30 + int((batch_num / total_batches) * 40)
            _cb("llm_batch", batch_pct, batch_num, total_batches)

            t0 = _time.time()
            results, err = _llm_annotate_batch_v2(annotated, batch, candidate_map)
            elapsed = round(_time.time() - t0, 1)
            logger.info(
                f"LLM v2 batch {batch_num}/{total_batches}: "
                f"{len(batch)} segs, {elapsed}s, {'OK' if err is None else 'FAIL'}"
            )

            if err is not None:
                llm_errors.append({
                    "batch_order_indices": err.batch_order_indices,
                    "exception_type": err.exception_type,
                    "message": err.message,
                    "raw_response_excerpt": err.raw_response_excerpt,
                })
                continue

            result_map = {r["order_index"]: r for r in results}
            for seg in batch:
                if seg.order_index in result_map:
                    r = result_map[seg.order_index]
                    # Resolve speaker from character_id → canonical name
                    speaker_id = r.get("speaker", "unknown")
                    entry = char_dict.find_by_id(speaker_id) if speaker_id != "unknown" else None
                    seg.predicted_speaker = entry.canonical_name if entry else speaker_id
                    seg.character_id = speaker_id if speaker_id != "unknown" else None
                    seg.confidence = float(r.get("confidence", seg.confidence))
                    seg.evidence_spans = r.get("evidence_spans", [])
                    seg.needs_review = bool(r.get("needs_review", False))
                    seg.reason = r.get("reason_short", seg.reason) + " (llm_v2)"
    else:
        logger.info("LLM disabled – skipping LLM v2 refinement")

    # ── Stage 5: global consistency ───────────────────────────────────────────
    _cb("consistency_pass", 75, 0, total_segs)
    try:
        apply_global_consistency(annotated, char_dict)
    except Exception as e:
        logger.warning(f"global_consistency failed (non-fatal): {e}")

    # ── Stage 6: speaker propagation ─────────────────────────────────────────
    _cb("speaker_propagation", 90, 0, total_segs)
    _maybe_apply_propagation(annotated, config)

    _cb("complete", 100, total_segs, total_segs)
    return annotated, llm_errors


def segment_speakers_llm_primary(
    raw_segments: List[RawSegment],
    config: Optional[DiarizationConfig] = None,
    known_characters: Optional[List[str]] = None,
    project_id: Optional[str] = None,
    db=None,
    progress_cb=None,
) -> tuple[List[AnnotatedSegment], List[dict]]:
    """LLM-primary chapter-unit speaker inference pipeline.

    This is a new mode that uses LLM as the primary speaker inference engine,
    processing text chapter by chapter for better context utilization.
    Falls back to rule-based if LLM is unavailable.
    """
    from app.orchestrator.services.llm_speaker_inference import run_llm_speaker_pipeline

    def _cb(stage: str, pct: int, cur: int = 0, total: int = 0) -> None:
        if progress_cb is not None:
            try:
                progress_cb(stage, pct, cur, total)
            except Exception:
                pass

    llm_errors: List[dict] = []

    # Reconstruct full text from raw segments
    full_text = "\n".join(seg.normalized_text or seg.raw_text for seg in raw_segments)

    # Extract known character names from segments if not provided
    if not known_characters:
        known_characters = list({
            seg.predicted_speaker
            for seg in raw_segments
            if hasattr(seg, "predicted_speaker")
            and seg.predicted_speaker not in ("unknown", "narrator", "protagonist", "")
        })

    _cb("llm_pipeline_start", 5, 0, len(raw_segments))

    try:
        def pipeline_progress(stage, detail):
            _cb(stage, 50, 0, len(raw_segments))

        segment_dicts = run_llm_speaker_pipeline(
            text=full_text,
            known_characters=known_characters,
            progress_callback=pipeline_progress,
        )

        # Convert dicts to AnnotatedSegment
        annotated: List[AnnotatedSegment] = []
        for sd in segment_dicts:
            annotated.append(AnnotatedSegment(
                chapter_index=sd.get("chapter_index", 0),
                order_index=sd.get("order_index", 0),
                raw_text=sd.get("raw_text", ""),
                normalized_text=sd.get("normalized_text", ""),
                segment_type=sd.get("segment_type", "narration"),
                predicted_speaker=sd.get("predicted_speaker", "unknown"),
                confidence=sd.get("confidence", 0.0),
                reason=sd.get("reason", ""),
                is_chapter_header=sd.get("is_chapter_header", False),
                candidates=sd.get("candidates", []),
                needs_review=sd.get("needs_review", False),
                evidence_spans=sd.get("evidence_spans", []),
            ))

        _cb("complete", 100, len(annotated), len(annotated))
        return annotated, llm_errors

    except Exception as e:
        logger.error(f"LLM-primary pipeline failed, falling back to enhanced: {e}")
        llm_errors.append({
            "batch_order_indices": [],
            "exception_type": type(e).__name__,
            "message": str(e),
            "raw_response_excerpt": "",
        })

        # Fallback to enhanced pipeline
        return segment_speakers_enhanced(
            raw_segments, config=config,
            project_id=project_id, db=db,
            progress_cb=progress_cb,
        )


def _make_enhanced_config(config: Optional[DiarizationConfig]) -> DiarizationConfig:
    """Return a config that has monologue_detection enabled (adds it if absent)."""
    if config is None:
        # Default config + monologue_detection
        rules = [RuleSpec(r) for r in DEFAULT_RULE_ORDER]
        rules.append(RuleSpec("monologue_detection", enabled=True))
        return DiarizationConfig(rules=rules)

    # Check if monologue_detection is already in the config
    ids = [r.rule_id for r in config.rules]
    if "monologue_detection" not in ids:
        new_rules = list(config.rules) + [RuleSpec("monologue_detection", enabled=True)]
        return DiarizationConfig(rules=new_rules, llm_fallback_mode=config.llm_fallback_mode)
    return config
