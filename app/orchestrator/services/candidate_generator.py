"""Candidate speaker generator.

For each dialogue / thought / unknown segment, generates an ordered list of
2–6 candidate speakers with per-candidate confidence priors and evidence
strings.  The LLM then re-ranks within these candidates.

Candidate sources (applied in order, merged, deduplicated, capped at 6):
  1. Preceding narration verb  – 「Xは言った」 immediately before
  2. First-person pronoun match – 僕/俺/私 → char with unambiguous mapping
  3. Turn alternation          – A→B→? → likely A in rapid back-and-forth
  4. Scene active characters   – chars who spoke in last N segments
  5. Speech style match        – 語尾パターンが辞書のキャラに一致
  6. Last speaker              – simple propagation fallback
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.orchestrator.services.character_builder import (
    ALL_FIRST_PERSON,
    SPEAKER_VERB_RE,
    CharacterDict,
    CharacterEntry,
    _STYLE_RES,
    _extract_dialogue_content,
)
from app.shared.logger import get_logger

logger = get_logger("candidate_generator")


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class CandidateSpeaker:
    name: str
    confidence: float              # prior 0.0–1.0 before LLM re-ranking
    evidence: List[str] = field(default_factory=list)  # human-readable reasons
    character_id: Optional[str] = None  # FK to character_master.id


@dataclass
class CandidateSet:
    segment_order_index: int
    candidates: List[CandidateSpeaker]
    scene_context: List[str]   # recent speaker names for LLM context


# ── Public API ────────────────────────────────────────────────────────────────

def generate_candidates(
    segment,            # AnnotatedSegment
    all_segments: list, # List[AnnotatedSegment]
    char_dict: CharacterDict,
    window: int = 10,
    max_candidates: int = 6,
) -> CandidateSet:
    """Generate and rank up to max_candidates speakers for a segment."""
    raw: List[CandidateSpeaker] = []

    raw += _from_preceding_narration(segment, all_segments, char_dict)
    raw += _from_pronoun_match(segment, char_dict)
    raw += _from_turn_alternation(segment, all_segments)
    raw += _from_scene_active_chars(segment, all_segments, char_dict, window)
    raw += _from_speech_style(segment, char_dict)
    raw += _from_last_speaker(segment, all_segments, char_dict)

    merged = _merge_and_rank(raw, max_candidates=max_candidates)

    # Scene context = speakers in last `window` segments
    scene_speakers = _recent_speakers(segment, all_segments, window)

    return CandidateSet(
        segment_order_index=segment.order_index,
        candidates=merged,
        scene_context=scene_speakers,
    )


def candidate_set_to_dicts(cs: CandidateSet) -> List[dict]:
    """Convert to JSON-serialisable list for DB storage and LLM prompt."""
    return [
        {
            "character_id": c.character_id or c.name,
            "name": c.name,
            "confidence": round(c.confidence, 3),
            "evidence": c.evidence,
        }
        for c in cs.candidates
    ]


# ── Source functions ──────────────────────────────────────────────────────────

def _from_preceding_narration(
    segment, all_segments: list, char_dict: CharacterDict,
) -> List[CandidateSpeaker]:
    """Extract speaker-verb pairs from narration segments immediately before."""
    results: List[CandidateSpeaker] = []
    narration_window = 3   # look at up to 3 narration segs before this one

    preceding = _preceding_narration_segs(segment, all_segments, narration_window)
    for narr_seg in preceding:
        text = narr_seg.normalized_text or narr_seg.raw_text
        for m in SPEAKER_VERB_RE.finditer(text):
            name = m.group(1).strip()
            verb = m.group(3)
            # confidence boost when the verb is in same sentence as the dialogue
            conf = 0.70
            evidence = f"「{name}{m.group(2)}{verb}」(直前地の文)"
            entry = char_dict.find(name)
            results.append(CandidateSpeaker(
                name=name,
                confidence=conf,
                evidence=[evidence],
                character_id=entry.id if entry else None,
            ))

    return results


def _from_pronoun_match(
    segment, char_dict: CharacterDict,
) -> List[CandidateSpeaker]:
    """Match first-person pronouns inside the segment text."""
    text = segment.normalized_text or segment.raw_text
    inner = _extract_dialogue_content(text) or text
    results: List[CandidateSpeaker] = []

    for pronoun in ALL_FIRST_PERSON:
        if pronoun in inner:
            canonical = char_dict.pronoun_map.get(pronoun)
            if canonical:
                entry = char_dict.find(canonical)
                results.append(CandidateSpeaker(
                    name=canonical,
                    confidence=0.50,
                    evidence=[f"一人称「{pronoun}」が辞書に一致"],
                    character_id=entry.id if entry else None,
                ))
            else:
                # Ambiguous pronoun — lower confidence for each char that uses it
                for entry in char_dict.entries:
                    if pronoun in entry.first_person_pronouns:
                        results.append(CandidateSpeaker(
                            name=entry.canonical_name,
                            confidence=0.25,
                            evidence=[f"一人称「{pronoun}」（複数候補あり）"],
                            character_id=entry.id,
                        ))
    return results


def _from_turn_alternation(
    segment, all_segments: list,
) -> List[CandidateSpeaker]:
    """In rapid back-and-forth A→B→? → next is likely A."""
    results: List[CandidateSpeaker] = []
    idx = segment.order_index

    # Find the 2 dialogue segments immediately before this one (skip narration)
    prev_dialogue: List[str] = []
    for seg in reversed(all_segments):
        if seg.order_index >= idx:
            continue
        if seg.segment_type in ("dialogue", "thought", "monologue"):
            sp = seg.predicted_speaker
            if sp not in ("unknown", "narrator", ""):
                prev_dialogue.append(sp)
                if len(prev_dialogue) == 2:
                    break

    if len(prev_dialogue) == 2 and prev_dialogue[0] != prev_dialogue[1]:
        # A→B pattern → next is likely A (prev_dialogue[1])
        likely = prev_dialogue[1]
        results.append(CandidateSpeaker(
            name=likely,
            confidence=0.45,
            evidence=[f"交互ターン (直前: {prev_dialogue[0]}→{prev_dialogue[1]})"],
        ))
    return results


def _from_scene_active_chars(
    segment, all_segments: list, char_dict: CharacterDict, window: int,
) -> List[CandidateSpeaker]:
    """Characters who appeared in the last `window` segments."""
    recent = _recent_speakers(segment, all_segments, window)
    results: List[CandidateSpeaker] = []
    for name in recent:
        if name in ("unknown", "narrator", ""):
            continue
        entry = char_dict.find(name)
        results.append(CandidateSpeaker(
            name=name,
            confidence=0.30,
            evidence=[f"シーン内登場 (直近{window}セグメント)"],
            character_id=entry.id if entry else None,
        ))
    return results


def _from_speech_style(
    segment, char_dict: CharacterDict,
) -> List[CandidateSpeaker]:
    """Match 語尾パターン from speech_style_hints."""
    text = segment.normalized_text or segment.raw_text
    inner = _extract_dialogue_content(text) or text
    results: List[CandidateSpeaker] = []

    if not inner:
        return results

    for style_re, style_label in _STYLE_RES:
        if style_re.search(inner):
            for entry in char_dict.entries:
                if style_label in entry.speech_style_hints:
                    results.append(CandidateSpeaker(
                        name=entry.canonical_name,
                        confidence=0.40,
                        evidence=[f"語尾パターン「{style_label}」が一致"],
                        character_id=entry.id,
                    ))
    return results


def _from_last_speaker(
    segment, all_segments: list, char_dict: CharacterDict,
) -> List[CandidateSpeaker]:
    """Simple fallback: last known speaker before this segment."""
    idx = segment.order_index
    for seg in reversed(all_segments):
        if seg.order_index >= idx:
            continue
        sp = seg.predicted_speaker
        if sp not in ("unknown", "narrator", ""):
            entry = char_dict.find(sp)
            return [CandidateSpeaker(
                name=sp,
                confidence=0.25,
                evidence=["直前話者（フォールバック）"],
                character_id=entry.id if entry else None,
            )]
    return []


# ── Merge and rank ────────────────────────────────────────────────────────────

def _merge_and_rank(
    raw: List[CandidateSpeaker],
    max_candidates: int = 6,
) -> List[CandidateSpeaker]:
    """Deduplicate by name (accumulate confidence, merge evidence), then sort."""
    merged: Dict[str, CandidateSpeaker] = {}
    for cand in raw:
        name = cand.name
        if name in merged:
            existing = merged[name]
            # Combine confidences with diminishing returns: c1 + c2*(1-c1)
            existing.confidence = min(
                1.0,
                existing.confidence + cand.confidence * (1.0 - existing.confidence),
            )
            for ev in cand.evidence:
                if ev not in existing.evidence:
                    existing.evidence.append(ev)
            if not existing.character_id and cand.character_id:
                existing.character_id = cand.character_id
        else:
            merged[name] = CandidateSpeaker(
                name=name,
                confidence=cand.confidence,
                evidence=list(cand.evidence),
                character_id=cand.character_id,
            )

    ranked = sorted(merged.values(), key=lambda c: c.confidence, reverse=True)
    return ranked[:max_candidates]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _preceding_narration_segs(segment, all_segments: list, n: int) -> list:
    """Return up to n narration segments immediately preceding this one."""
    idx = segment.order_index
    result = []
    for seg in reversed(all_segments):
        if seg.order_index >= idx:
            continue
        if seg.segment_type in ("narration",):
            result.append(seg)
            if len(result) >= n:
                break
    return list(reversed(result))


def _recent_speakers(segment, all_segments: list, window: int) -> List[str]:
    """Return unique speakers in the last `window` segments before this one."""
    idx = segment.order_index
    seen = []
    for seg in reversed(all_segments):
        if seg.order_index >= idx:
            continue
        if len(seen) >= window:
            break
        sp = seg.predicted_speaker
        if sp and sp not in seen:
            seen.append(sp)
    return seen
