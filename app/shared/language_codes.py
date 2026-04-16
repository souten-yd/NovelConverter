"""Normalize language identifiers for the Qwen3-TTS backend.

Qwen3-TTS expects full English language names (e.g. ``japanese``) rather than
ISO 639-1 codes (``ja``). Historically the project stored short codes in the
database, voice profiles, and UI, so this helper maps them to the full names
the backend accepts. Unknown values pass through unchanged so that explicit
supported names (``japanese``, ``auto``, ...) also work.
"""
from __future__ import annotations

from typing import Optional

_SHORT_TO_FULL = {
    "ja": "japanese",
    "jp": "japanese",
    "jpn": "japanese",
    "en": "english",
    "eng": "english",
    "zh": "chinese",
    "zh-cn": "chinese",
    "zh_cn": "chinese",
    "cn": "chinese",
    "ko": "korean",
    "kor": "korean",
    "es": "spanish",
    "spa": "spanish",
    "fr": "french",
    "fra": "french",
    "fre": "french",
    "de": "german",
    "ger": "german",
    "deu": "german",
    "it": "italian",
    "ita": "italian",
    "pt": "portuguese",
    "por": "portuguese",
    "ru": "russian",
    "rus": "russian",
}

DEFAULT_TTS_LANGUAGE = "japanese"


def normalize_tts_language(language: Optional[str], default: str = DEFAULT_TTS_LANGUAGE) -> str:
    """Return the Qwen3-TTS language name for ``language``.

    - ``None`` / empty string → ``default`` (``japanese`` by default).
    - Known short codes (``ja``, ``en``, ``zh`` …) are mapped to full names.
    - Any other value is lower-cased and returned as-is so that explicit
      supported names (``japanese``, ``auto`` …) pass through untouched.
    """
    if not language:
        return default
    key = language.strip().lower()
    if not key:
        return default
    return _SHORT_TO_FULL.get(key, key)
