"""Character dictionary builder for light-novel speaker identification.

Scans annotated segments (especially narration) for:
  - Speaker-verb pairs  (「Xは言った」「Xが答えた」)
  - Self-introductions  (「私はXです」「僕はXだ」)
  - Address terms       (「Xさん、〜」)
  - First-person pronoun occurrences inside dialogue

Produces an in-memory CharacterDict and helpers for persisting to the DB.
"""
from __future__ import annotations

import re
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.shared.logger import get_logger

logger = get_logger("character_builder")

# ── Regex patterns ────────────────────────────────────────────────────────────

# 「Xは言った」「Xが答えた」 etc. – speaker is up to 10 chars before は/が + verb
SPEAKER_VERB_RE = re.compile(
    r'([^\s「」『』。、\n]{1,10})(は|が)'
    r'(言った|答えた|叫んだ|呟いた|続けた|笑った|怒った|泣いた|'
    r'聞いた|囁いた|怒鳴った|告げた|返した|応えた|付け加えた|'
    r'呟く|言う|答える|叫ぶ|続ける|笑う|怒る|泣く)',
)

# 「Xさん」「Xちゃん」「X君」「X様」 in any context
HONORIFIC_RE = re.compile(
    r'([^\s「」『』。、\n]{1,8})(さん|ちゃん|くん|君|様|殿|先生|先輩|後輩)',
)

# Self-intro: 「私はXです」「僕はXだ」「俺はXだぞ」
SELF_INTRO_RE = re.compile(
    r'(私|僕|俺|わたし|ぼく|おれ|あたし|うち)は([^\s「」。、]{1,10})'
    r'(です|だ|だよ|だぞ|だぜ|ですよ|だよね)',
)

# First-person pronouns
FIRST_PERSON_PRONOUNS = {
    "male": ["僕", "俺", "おれ", "ぼく", "わし", "拙者"],
    "female": ["私", "あたし", "うち", "あーし"],
    "neutral": ["わたし", "自分", "小生"],
}
ALL_FIRST_PERSON = {p for pronouns in FIRST_PERSON_PRONOUNS.values() for p in pronouns}

# Speech-ending patterns by character archetype
SPEECH_STYLE_PATTERNS: List[Tuple[str, str]] = [
    # pattern suffix → hint label
    (r"だよ$|だよね$|だよ？$", "だよ系"),
    (r"ですわ$|ますわ$|ですわよ$", "ですわ系"),
    (r"じゃ$|じゃな$|じゃぞ$|じゃろ$", "じゃ系"),
    (r"だぞ$|だぜ$|だろ$", "だぞ系"),
    (r"ね$|ねえ$|ねぇ$", "ね系"),
    (r"のよ$|のね$|わよ$|わね$|かしら$", "わよ系"),
    (r"ます$|ません$|ました$|ますよ$", "ます系"),
]
_STYLE_RES = [(re.compile(p), label) for p, label in SPEECH_STYLE_PATTERNS]


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class CharacterEntry:
    canonical_name: str
    aliases: List[str] = field(default_factory=list)
    first_person_pronouns: List[str] = field(default_factory=list)
    speech_style_hints: List[str] = field(default_factory=list)
    honorifics_used: List[str] = field(default_factory=list)  # chars this char calls others with honorific
    gender_hint: str = "unknown"   # "male" | "female" | "unknown"
    role_hint: str = "unknown"     # "protagonist" | "heroine" | "support" | "unknown"
    evidence_count: int = 0        # times seen in text
    _id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def id(self) -> str:
        return self._id

    def all_names(self) -> List[str]:
        return [self.canonical_name] + self.aliases


@dataclass
class CharacterDict:
    entries: List[CharacterEntry] = field(default_factory=list)
    # alias/name → index into entries
    _name_to_idx: Dict[str, int] = field(default_factory=dict, repr=False)
    # unambiguous pronoun → canonical_name (only set if single char uses it)
    pronoun_map: Dict[str, str] = field(default_factory=dict)

    def rebuild_index(self) -> None:
        self._name_to_idx = {}
        for i, entry in enumerate(self.entries):
            for name in entry.all_names():
                self._name_to_idx[name] = i

    def find(self, name: str) -> Optional[CharacterEntry]:
        idx = self._name_to_idx.get(name)
        return self.entries[idx] if idx is not None else None

    def find_by_id(self, char_id: str) -> Optional[CharacterEntry]:
        for e in self.entries:
            if e.id == char_id:
                return e
        return None

    def all_names(self) -> List[str]:
        names: List[str] = []
        for e in self.entries:
            names.extend(e.all_names())
        return names


# ── Main builder ──────────────────────────────────────────────────────────────

def build_character_dict(
    segments: list,  # List[AnnotatedSegment] — imported lazily to avoid circular
    existing_speaker_names: Optional[List[str]] = None,
) -> CharacterDict:
    """Build an in-memory CharacterDict by scanning narration/dialogue segments.

    Parameters
    ----------
    segments:
        AnnotatedSegment list (from rule_based_pass or post-LLM).
    existing_speaker_names:
        Pre-known speaker names (e.g. from previous manual mapping).
        These are seeded as CharacterEntry with empty metadata.
    """
    name_counter: Counter = Counter()
    pronoun_counter: Dict[str, Counter] = defaultdict(Counter)  # name → pronoun counts
    style_counter: Dict[str, Counter] = defaultdict(Counter)    # name → style counts

    for seg in segments:
        text = seg.normalized_text or seg.raw_text

        # ── Speaker-verb pairs from narration ────────────────────────────────
        if seg.segment_type in ("narration", "unknown"):
            for m in SPEAKER_VERB_RE.finditer(text):
                name = m.group(1).strip()
                if _is_valid_name(name):
                    name_counter[name] += 2  # direct verb evidence = double weight

        # ── Honorific address terms ───────────────────────────────────────────
        for m in HONORIFIC_RE.finditer(text):
            name = m.group(1).strip()
            if _is_valid_name(name):
                name_counter[name] += 1

        # ── Self-introductions inside dialogue ───────────────────────────────
        if seg.segment_type == "dialogue":
            for m in SELF_INTRO_RE.finditer(text):
                # m.group(2) is the name being introduced
                intro_name = m.group(2).strip()
                if _is_valid_name(intro_name) and seg.predicted_speaker not in ("unknown", "narrator", ""):
                    # The speaker introduced themselves
                    if seg.predicted_speaker != intro_name:
                        # Track as alias
                        name_counter[intro_name] += 3
                        name_counter[seg.predicted_speaker] += 1

        # ── Pronoun evidence inside dialogue ─────────────────────────────────
        if seg.segment_type in ("dialogue", "thought", "monologue"):
            speaker = seg.predicted_speaker
            if speaker not in ("unknown", "narrator", ""):
                for pronoun in ALL_FIRST_PERSON:
                    if pronoun in text:
                        pronoun_counter[speaker][pronoun] += 1

            # Speech style detection
            for style_re, style_label in _STYLE_RES:
                inner = _extract_dialogue_content(text)
                if inner and style_re.search(inner):
                    if speaker not in ("unknown", "narrator", ""):
                        style_counter[speaker][style_label] += 1

    # ── Seed from existing speakers ──────────────────────────────────────────
    if existing_speaker_names:
        for name in existing_speaker_names:
            if name not in ("unknown", "narrator", "") and name not in name_counter:
                name_counter[name] += 1

    # ── Build entries ─────────────────────────────────────────────────────────
    entries = _build_entries(name_counter, pronoun_counter, style_counter)
    entries = _merge_aliases(entries)

    char_dict = CharacterDict(entries=entries)
    char_dict.rebuild_index()
    char_dict.pronoun_map = _build_pronoun_map(entries)

    logger.info(
        f"character_builder: {len(entries)} characters found, "
        f"pronoun_map={len(char_dict.pronoun_map)} entries"
    )
    return char_dict


# ── Helpers ───────────────────────────────────────────────────────────────────

_INVALID_NAME_PREFIXES = ("と", "が", "は", "を", "に", "で", "も", "の", "や", "へ", "から", "まで")
_MIN_NAME_LEN = 1
_MAX_NAME_LEN = 10
_STOPWORDS = frozenset({
    "narrator", "unknown", "みんな", "全員", "誰", "誰か", "彼", "彼女",
    "彼ら", "それ", "あれ", "これ", "その", "あの", "この",
})


def _is_valid_name(name: str) -> bool:
    if not name or len(name) < _MIN_NAME_LEN or len(name) > _MAX_NAME_LEN:
        return False
    if name in _STOPWORDS:
        return False
    if name in ALL_FIRST_PERSON:
        return False
    for prefix in _INVALID_NAME_PREFIXES:
        if name.startswith(prefix):
            return False
    # Must contain at least one CJK/kana character
    if not re.search(r'[一-龥ぁ-んァ-ン]', name):
        return False
    return True


def _extract_dialogue_content(text: str) -> Optional[str]:
    """Extract text inside 「」 or 『』 brackets."""
    m = re.search(r'[「『]([^」』]+)[」』]', text)
    return m.group(1) if m else None


def _build_entries(
    name_counter: Counter,
    pronoun_counter: Dict[str, Counter],
    style_counter: Dict[str, Counter],
) -> List[CharacterEntry]:
    entries: List[CharacterEntry] = []
    for name, count in name_counter.most_common():
        if count < 1:
            continue
        # Infer gender from pronouns
        pronouns = list(pronoun_counter.get(name, {}).keys())
        gender = _infer_gender(pronouns)

        # Top speech styles
        styles = [s for s, _ in style_counter.get(name, Counter()).most_common(3)]

        entry = CharacterEntry(
            canonical_name=name,
            first_person_pronouns=pronouns,
            speech_style_hints=styles,
            gender_hint=gender,
            evidence_count=count,
        )
        entries.append(entry)
    return entries


def _infer_gender(pronouns: List[str]) -> str:
    male_hits = sum(1 for p in pronouns if p in FIRST_PERSON_PRONOUNS["male"])
    female_hits = sum(1 for p in pronouns if p in FIRST_PERSON_PRONOUNS["female"])
    if male_hits > female_hits:
        return "male"
    if female_hits > male_hits:
        return "female"
    return "unknown"


def _merge_aliases(entries: List[CharacterEntry]) -> List[CharacterEntry]:
    """Cluster entries that are likely the same person using name prefix heuristics.

    Simple rule:  if name A is a prefix of name B and len(A) >= 2, merge B into A
    (canonical name stays A = the shorter, more common form).
    """
    if len(entries) <= 1:
        return entries

    # Sort by evidence count descending so dominant names win
    entries = sorted(entries, key=lambda e: e.evidence_count, reverse=True)
    merged: List[CharacterEntry] = []
    absorbed: set = set()

    for i, entry in enumerate(entries):
        if i in absorbed:
            continue
        for j, other in enumerate(entries):
            if j <= i or j in absorbed:
                continue
            # Prefix match (name A is prefix of name B or vice-versa)
            a, b = entry.canonical_name, other.canonical_name
            if len(a) >= 2 and (b.startswith(a) or a.startswith(b)):
                # Absorb the shorter evidence count into the higher one
                entry.aliases.append(b if b != a else a)
                entry.evidence_count += other.evidence_count
                absorbed.add(j)
                logger.debug(f"character_builder: merged '{b}' → '{a}'")
        merged.append(entry)

    return merged


def _build_pronoun_map(entries: List[CharacterEntry]) -> Dict[str, str]:
    """Map pronoun → canonical_name only when it is unambiguous (one char uses it)."""
    pronoun_to_names: Dict[str, List[str]] = defaultdict(list)
    for entry in entries:
        for pronoun in entry.first_person_pronouns:
            pronoun_to_names[pronoun].append(entry.canonical_name)

    return {
        pronoun: names[0]
        for pronoun, names in pronoun_to_names.items()
        if len(names) == 1
    }


# ── DB persistence helpers ────────────────────────────────────────────────────

def persist_character_dict(
    char_dict: CharacterDict,
    project_id: str,
    db,  # SQLAlchemy Session
) -> None:
    """Save CharacterDict to character_master + character_aliases tables.

    Replaces any existing entries for this project.
    """
    from app.shared.models import CharacterMaster, CharacterAlias

    # Clear existing
    db.query(CharacterAlias).filter(CharacterAlias.project_id == project_id).delete()
    db.query(CharacterMaster).filter(CharacterMaster.project_id == project_id).delete()
    db.flush()

    for entry in char_dict.entries:
        cm = CharacterMaster(
            id=entry.id,
            project_id=project_id,
            canonical_name=entry.canonical_name,
            aliases=entry.aliases,
            first_person_pronouns=entry.first_person_pronouns,
            speech_style_hints=entry.speech_style_hints,
            honorifics_used=entry.honorifics_used,
            gender_hint=entry.gender_hint,
            role_hint=entry.role_hint,
            segment_count=entry.evidence_count,
        )
        db.add(cm)

        for alias in entry.aliases:
            ca = CharacterAlias(
                character_id=entry.id,
                project_id=project_id,
                alias=alias,
                alias_type="name",
            )
            db.add(ca)

        for pronoun in entry.first_person_pronouns:
            ca = CharacterAlias(
                character_id=entry.id,
                project_id=project_id,
                alias=pronoun,
                alias_type="pronoun",
            )
            db.add(ca)

    db.flush()
    logger.info(
        f"persist_character_dict: saved {len(char_dict.entries)} characters "
        f"for project={project_id}"
    )


def load_character_dict(project_id: str, db) -> Optional[CharacterDict]:
    """Load CharacterDict from DB.  Returns None if no entries exist."""
    from app.shared.models import CharacterMaster

    rows = db.query(CharacterMaster).filter(CharacterMaster.project_id == project_id).all()
    if not rows:
        return None

    entries: List[CharacterEntry] = []
    for row in rows:
        entry = CharacterEntry(
            canonical_name=row.canonical_name,
            aliases=list(row.aliases or []),
            first_person_pronouns=list(row.first_person_pronouns or []),
            speech_style_hints=list(row.speech_style_hints or []),
            honorifics_used=list(row.honorifics_used or []),
            gender_hint=row.gender_hint or "unknown",
            role_hint=row.role_hint or "unknown",
            evidence_count=row.segment_count or 0,
            _id=row.id,
        )
        entries.append(entry)

    char_dict = CharacterDict(entries=entries)
    char_dict.rebuild_index()
    char_dict.pronoun_map = _build_pronoun_map(entries)
    return char_dict
