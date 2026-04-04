"""Japanese light-novel text normalizer.

Applies a chain of rule-based transformations to raw OCR/input text and
records before/after diffs for each rule that fires.  The result is used
both for cleaner segmentation and for persisting NormalizationLog rows.

Rules (applied in order):
  1. quotemark_unify   – 全角/異体引用符を 「」『』 に統一
  2. ellipsis_dash_norm – 三点リーダ・ダッシュ・長音の正規化
  3. ruby_removal      – ルビ（《》｜ 形式・HTMLタグ）除去
  4. ocr_noise_removal – OCRゴミ行（孤立1文字・罫線記号）除去
  5. page_join         – ページ区切りマーカー周辺のセリフ連結
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from app.shared.logger import get_logger

logger = get_logger("text_normalizer")


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class NormalizationEntry:
    rule_name: str
    before: str   # text before this rule (may be many chars; caller should trim)
    after: str    # text after this rule


@dataclass
class NormalizationResult:
    text: str
    logs: List[NormalizationEntry] = field(default_factory=list)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _apply_rule(
    text: str,
    rule_name: str,
    transform: Callable[[str], str],
    logs: List[NormalizationEntry],
) -> str:
    """Apply transform, record log entry only when text actually changes."""
    result = transform(text)
    if result != text:
        logs.append(NormalizationEntry(
            rule_name=rule_name,
            before=text[:200],
            after=result[:200],
        ))
        logger.debug(f"norm[{rule_name}]: {len(text)}→{len(result)} chars")
    return result


# ── Rule 1: 引用符統一 ────────────────────────────────────────────────────────

# Half-width/variant bracket pairs → full-width Japanese
_QUOTE_REPLACEMENTS = [
    # Variant opening/closing single curly quotes → 「」
    ("\u2018", "「"), ("\u2019", "」"),  # ' '
    # Variant double curly quotes → 『』
    ("\u201c", "『"), ("\u201d", "』"),  # " "
    # Half-width corner brackets
    ("\uff62", "「"), ("\uff63", "」"),  # ｢ ｣
    # Low-9 quotation marks (OCR artefact)
    ("\u201a", "、"), ("\u201e", "、"),
    # Corner brackets already correct – no-op entries removed
]

_FULLWIDTH_OPEN = re.compile(r'[\u300c\u300e]')   # 「『
_FULLWIDTH_CLOSE = re.compile(r'[\u300d\u300f]')  # 」』


def _unify_quote_marks(text: str) -> str:
    for src, dst in _QUOTE_REPLACEMENTS:
        text = text.replace(src, dst)
    return text


# ── Rule 2: 三点リーダ・ダッシュ・長音正規化 ─────────────────────────────────

# Repeated ellipsis → single …  (but keep 2× → ……)
_TRIPLE_ELLIPSIS = re.compile(r'\.{3,}|…{3,}')
_DOUBLE_ELLIPSIS = re.compile(r'…{2}')   # keep 2× as is

# Repeated dash (2+ em-dashes or double-hyphen) → ——
_LONG_DASH = re.compile(r'[—―\-]{2,}')
# Repeated wave dash
_WAVE_DASH = re.compile(r'[〜～]{2,}')
# 3+ repeated 長音 → 2
_LONG_VOWEL = re.compile(r'ー{3,}')


def _normalize_ellipsis_dash(text: str) -> str:
    # Normalise 3+ dots to … (but leave 2× … alone)
    text = _TRIPLE_ELLIPSIS.sub("……", text)
    # Repeated dashes → —
    text = _LONG_DASH.sub("——", text)
    # Repeated wave dash → 〜
    text = _WAVE_DASH.sub("〜", text)
    # Trim excessive 長音
    text = _LONG_VOWEL.sub("ーー", text)
    return text


# ── Rule 3: ルビ除去 ──────────────────────────────────────────────────────────

# 《》inline ruby: 漢字《よみ》 → 漢字
_RUBY_INLINE = re.compile(r'(?<=[^\s｜])《[^》]+》')
# ｜ explicit ruby: ｜漢字《よみ》 → 漢字
_RUBY_EXPLICIT = re.compile(r'｜([^《\s]+)《[^》]+》')
# 傍点: 《《text》》or ｜text《・・・》
_RUBY_DOTS = re.compile(r'《[・。、，．…]+》')
# HTML ruby: <ruby>漢字<rt>よみ</rt></ruby>
_RUBY_HTML = re.compile(r'<ruby>([^<]+)<rt>[^<]*</rt></ruby>', re.IGNORECASE)
_RUBY_HTML2 = re.compile(r'<rp>[^<]*</rp>', re.IGNORECASE)
# Bracketed reading annotation: 漢字(よみ) only when hiragana/katakana inside
_RUBY_PAREN = re.compile(r'([一-龥ぁ-ん]{1,8})[(（]([ぁ-んァ-ン]{1,8})[)）]')


def _remove_ruby(text: str) -> str:
    text = _RUBY_HTML.sub(r'\1', text)
    text = _RUBY_HTML2.sub('', text)
    text = _RUBY_DOTS.sub('', text)
    text = _RUBY_EXPLICIT.sub(r'\1', text)
    text = _RUBY_INLINE.sub('', text)
    text = _RUBY_PAREN.sub(r'\1', text)
    return text


# ── Rule 4: OCRノイズ除去 ─────────────────────────────────────────────────────

# Lines that are just a single CJK or ASCII symbol (OCR artefact)
_LONE_SYMBOL_LINE = re.compile(
    r'^[^\S\n]*[！-／：-＠［-｀｛-～\|｜…‥・。、\-=_~@#$%^&*()\[\]{}/\\]{1,3}[^\S\n]*$',
    re.MULTILINE,
)
# Lines of only box-drawing / separator characters (─━═□■●○◎◆◇★☆▲▼▶◀)
_SEPARATOR_LINE = re.compile(
    r'^[^\S\n]*[─━═\-=*_◆◇■□●○◎★☆▲▼▶◀]{3,}[^\S\n]*$',
    re.MULTILINE,
)
# Stray lone single kanji/kana on its own line (likely OCR noise from furigana)
_LONE_SINGLE_CHAR = re.compile(r'^[^\S\n]*[一-龥ぁ-んァ-ン]{1}[^\S\n]*$', re.MULTILINE)
# Page marker injected by OCR engine (e.g. "=== page3.jpg ===")
_PAGE_MARKER = re.compile(r'^={3,}[^\n]*\.(jpg|png|jpeg|webp|gif)[^\n]*={0,3}$', re.MULTILINE | re.IGNORECASE)


def _remove_ocr_noise(text: str) -> str:
    text = _PAGE_MARKER.sub('\n', text)
    text = _SEPARATOR_LINE.sub('', text)
    text = _LONE_SYMBOL_LINE.sub('', text)
    text = _LONE_SINGLE_CHAR.sub('', text)
    # collapse resulting multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


# ── Rule 5: ページまたぎのセリフ結合 ─────────────────────────────────────────

# Detect an unclosed 「 that immediately precedes a page-marker line,
# then connect to the continuation on the next non-empty line.
_PAGE_SPLIT_DIALOGUE = re.compile(
    r'(「[^」\n]+)\n+'          # opening bracket fragment (no close on same line)
    r'(?:={3,}[^\n]*\n+)?'      # optional OCR page marker (already stripped above)
    r'([^「」\n]*」)',            # continuation ending with close bracket
    re.DOTALL,
)


def _join_page_crossing_dialogue(text: str) -> str:
    """Join dialogue that was split across two OCR page boundaries."""
    return _PAGE_SPLIT_DIALOGUE.sub(r'\1\2', text)


# ── Public API ────────────────────────────────────────────────────────────────

_RULES: List[Tuple[str, Callable[[str], str]]] = [
    ("quotemark_unify",   _unify_quote_marks),
    ("ellipsis_dash_norm", _normalize_ellipsis_dash),
    ("ruby_removal",       _remove_ruby),
    ("ocr_noise_removal",  _remove_ocr_noise),
    ("page_join",          _join_page_crossing_dialogue),
]


def normalize_japanese_text(raw_text: str) -> NormalizationResult:
    """Apply all normalisation rules in sequence.

    Returns NormalizationResult with the cleaned text and a log of every
    rule that produced a change (for NormalizationLog persistence).
    """
    text = raw_text
    logs: List[NormalizationEntry] = []

    for rule_name, transform in _RULES:
        text = _apply_rule(text, rule_name, transform, logs)

    if logs:
        logger.info(
            f"text_normalizer: {len(logs)} rule(s) fired, "
            f"{len(raw_text)}→{len(text)} chars"
        )

    return NormalizationResult(text=text, logs=logs)
