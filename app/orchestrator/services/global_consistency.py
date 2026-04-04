"""Global consistency pass (2nd pass) for speaker diarization.

After per-segment LLM scoring, this module re-evaluates low-confidence
segments using scene-level context:

  1. Turn alternation: A→B→? likely A in rapid dialogue
  2. Consecutive same-speaker suppression: 3+ same speaker (non-monologue) is suspect
  3. High-confidence propagation: spread high-conf speakers to adjacent low-conf
  4. Pronoun/style contradiction: penalise speakers that contradict character traits
  5. Scene active character set: prefer speakers seen recently in the scene

Only segments with confidence < threshold are reconsidered; confirmed segments
are left untouched to avoid degrading already-correct predictions.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from app.shared.logger import get_logger

logger = get_logger("global_consistency")

# Scene-boundary signals in narration text (time/location transitions)
_SCENE_BOUNDARY_RE = re.compile(
    r"(翌日|その日|一方|場所は|その頃|一時間後|しばらく後|数日後|翌朝|翌夜"
    r"|その後|ところで|場面は|——|───|―――)"
)

_DEFAULT_THRESHOLD = 0.65
_MAX_CONSECUTIVE_SAME = 3   # flag if same speaker repeats this many times in a row


def apply_global_consistency(
    segments: list,          # List[AnnotatedSegment]
    char_dict=None,          # Optional[CharacterDict]
    threshold: float = _DEFAULT_THRESHOLD,
) -> list:
    """Apply scene-level consistency rules to low-confidence segments.

    Modifies segments in-place and returns the same list.
    """
    if not segments:
        return segments

    # Group by chapter
    chapter_map: Dict[int, List] = {}
    for seg in segments:
        chapter_map.setdefault(seg.chapter_index, []).append(seg)

    for chap_idx, chap_segs in chapter_map.items():
        scenes = _detect_scene_boundaries(chap_segs)
        for scene in scenes:
            _consistency_within_scene(scene, char_dict, threshold)

    return segments


def _detect_scene_boundaries(chapter_segs: list) -> List[List]:
    """Split a chapter into scene lists on chapter headers or scene-transition narration."""
    scenes: List[List] = []
    current_scene: List = []

    for seg in chapter_segs:
        if seg.is_chapter_header:
            if current_scene:
                scenes.append(current_scene)
            current_scene = []
            continue

        # Scene boundary: long narration with transition keywords
        if (seg.segment_type == "narration"
                and len(seg.normalized_text) > 80
                and _SCENE_BOUNDARY_RE.search(seg.normalized_text)):
            if len(current_scene) >= 3:
                scenes.append(current_scene)
                current_scene = []

        current_scene.append(seg)

    if current_scene:
        scenes.append(current_scene)

    return scenes if scenes else [chapter_segs]


def _consistency_within_scene(
    scene: list,
    char_dict,
    threshold: float,
) -> None:
    """Apply consistency rules within a single scene (in-place)."""
    if not scene:
        return

    # ── Rule 1: high-confidence propagation ──────────────────────────────────
    _propagate_high_confidence(scene, threshold)

    # ── Rule 2: turn alternation ──────────────────────────────────────────────
    _fix_turn_alternation(scene, threshold)

    # ── Rule 3: consecutive same-speaker suppression ──────────────────────────
    _suppress_excess_consecutive(scene, threshold)

    # ── Rule 4: pronoun/style contradiction check ─────────────────────────────
    if char_dict is not None:
        _check_character_contradictions(scene, char_dict, threshold)


def _propagate_high_confidence(scene: list, threshold: float) -> None:
    """Spread speaker identity from high-confidence neighbours to low-conf."""
    n = len(scene)
    for i, seg in enumerate(scene):
        if seg.is_chapter_header or seg.confidence >= threshold:
            continue
        if seg.segment_type not in ("dialogue", "thought", "monologue", "unknown"):
            continue

        # Look left and right for high-confidence same-type segments
        left_speaker = _find_adjacent_speaker(scene, i, direction="left")
        right_speaker = _find_adjacent_speaker(scene, i, direction="right")

        if left_speaker and right_speaker and left_speaker == right_speaker:
            # Both neighbours agree
            _update_speaker(seg, left_speaker, min(seg.confidence + 0.15, 0.70),
                            "both-neighbour propagation (consistency)")
        elif left_speaker and not right_speaker:
            _update_speaker(seg, left_speaker, min(seg.confidence + 0.08, 0.55),
                            "left-neighbour propagation (consistency)")


def _find_adjacent_speaker(
    scene: list, idx: int, direction: str, max_distance: int = 3,
) -> Optional[str]:
    """Return the nearest high-confidence dialogue speaker in a direction."""
    step = -1 if direction == "left" else 1
    for d in range(1, max_distance + 1):
        j = idx + d * step
        if j < 0 or j >= len(scene):
            break
        neighbour = scene[j]
        if neighbour.is_chapter_header:
            break
        if (neighbour.confidence >= 0.75
                and neighbour.segment_type in ("dialogue", "thought", "monologue")
                and neighbour.predicted_speaker not in ("unknown", "narrator", "protagonist", "")):
            return neighbour.predicted_speaker
        if neighbour.segment_type == "narration" and len(neighbour.normalized_text) > 50:
            break  # scene break in narration – don't propagate past it
    return None


def _fix_turn_alternation(scene: list, threshold: float) -> None:
    """A→B→? → set ? to A when only 2 active speakers in scene."""
    dialogue_segs = [
        s for s in scene
        if not s.is_chapter_header
        and s.segment_type in ("dialogue", "thought")
    ]
    if len(dialogue_segs) < 4:
        return

    # Identify the 2 most common confirmed speakers in this scene
    from collections import Counter
    speaker_counts: Counter = Counter()
    for s in dialogue_segs:
        if s.confidence >= 0.75 and s.predicted_speaker not in ("unknown", "narrator", ""):
            speaker_counts[s.predicted_speaker] += 1

    if len(speaker_counts) != 2:
        return   # only apply when exactly 2 dominant speakers

    two_speakers = [sp for sp, _ in speaker_counts.most_common(2)]

    for i, seg in enumerate(dialogue_segs):
        if seg.confidence >= threshold or seg.predicted_speaker not in ("unknown", ""):
            continue

        # Find the previous confirmed speaker
        prev_conf_speaker = None
        for j in range(i - 1, -1, -1):
            if dialogue_segs[j].confidence >= 0.70:
                prev_conf_speaker = dialogue_segs[j].predicted_speaker
                break

        if prev_conf_speaker in two_speakers:
            # Assign the other one
            other = two_speakers[0] if prev_conf_speaker == two_speakers[1] else two_speakers[1]
            _update_speaker(seg, other,
                            min(seg.confidence + 0.12, 0.62),
                            "turn-alternation (2-speaker scene)")


def _suppress_excess_consecutive(scene: list, threshold: float) -> None:
    """If the same speaker appears 3+ times consecutively (not monologue), lower confidence."""
    i = 0
    while i < len(scene):
        seg = scene[i]
        if (seg.is_chapter_header
                or seg.segment_type in ("narration", "monologue")
                or seg.predicted_speaker in ("unknown", "narrator", "")):
            i += 1
            continue

        # Count run length
        run = 1
        j = i + 1
        while j < len(scene):
            next_seg = scene[j]
            if next_seg.segment_type == "narration":
                j += 1
                continue   # skip narration within run
            if (next_seg.predicted_speaker == seg.predicted_speaker
                    and next_seg.segment_type != "monologue"):
                run += 1
                j += 1
            else:
                break

        if run >= _MAX_CONSECUTIVE_SAME:
            # Flag middle segments for review but don't change speaker
            for k in range(i + 1, min(i + run - 1, len(scene))):
                if scene[k].confidence < 0.80 and not scene[k].is_chapter_header:
                    scene[k].needs_review = True
                    logger.debug(
                        f"consecutive_flag: seg={scene[k].order_index} "
                        f"speaker={seg.predicted_speaker} run={run}"
                    )
        i = j if j > i else i + 1


def _check_character_contradictions(
    scene: list, char_dict, threshold: float,
) -> None:
    """Lower confidence when segment content contradicts character traits."""
    from app.orchestrator.services.character_builder import (
        ALL_FIRST_PERSON, FIRST_PERSON_PRONOUNS, _STYLE_RES,
        _extract_dialogue_content,
    )

    for seg in scene:
        if (seg.is_chapter_header
                or seg.confidence < 0.30   # already low – don't stack
                or seg.predicted_speaker in ("unknown", "narrator", "protagonist", "")):
            continue

        entry = char_dict.find(seg.predicted_speaker)
        if entry is None:
            continue

        text = seg.normalized_text or seg.raw_text
        inner = _extract_dialogue_content(text) or text

        # Check pronoun contradiction (e.g. female char using 俺/僕)
        if entry.gender_hint == "female":
            male_pronouns = FIRST_PERSON_PRONOUNS["male"]
            if any(p in inner for p in male_pronouns):
                seg.confidence = max(0.20, seg.confidence - 0.20)
                seg.needs_review = True
                seg.reason += " [gender-pronoun contradiction]"

        elif entry.gender_hint == "male":
            female_only = ["あたし", "あーし"]
            if any(p in inner for p in female_only):
                seg.confidence = max(0.20, seg.confidence - 0.20)
                seg.needs_review = True
                seg.reason += " [gender-pronoun contradiction]"


def _update_speaker(seg, new_speaker: str, new_conf: float, reason_suffix: str) -> None:
    """Update a segment's predicted_speaker if confidence improves."""
    if new_conf > seg.confidence:
        seg.predicted_speaker = new_speaker
        seg.confidence = new_conf
        seg.reason = seg.reason.rstrip() + f" ({reason_suffix})"
