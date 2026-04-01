"""Text ingestion layer – reads uploaded text files."""
from __future__ import annotations

import re
from pathlib import Path

from app.shared.logger import get_logger

logger = get_logger("ingest")

# BOM variants
_BOM_MAP = {
    b"\xef\xbb\xbf": "utf-8-sig",
    b"\xff\xfe": "utf-16-le",
    b"\xfe\xff": "utf-16-be",
}


def detect_encoding(raw: bytes) -> str:
    for bom, enc in _BOM_MAP.items():
        if raw.startswith(bom):
            return enc
    return "utf-8"


def load_text_file(path: Path) -> str:
    """Load a text file, handling common encodings and BOM."""
    raw = path.read_bytes()
    encoding = detect_encoding(raw)
    try:
        text = raw.decode(encoding)
    except UnicodeDecodeError:
        # fallback
        text = raw.decode("cp932", errors="replace")
        logger.warning(f"Fell back to cp932 for {path}")
    # strip BOM char if present
    text = text.lstrip("\ufeff")
    logger.info(f"Loaded {path} ({len(text)} chars, encoding={encoding})")
    return text


def save_project_text(project_dir: Path, text: str, filename: str = "original.txt") -> Path:
    """Persist raw uploaded text to project directory."""
    project_dir.mkdir(parents=True, exist_ok=True)
    dest = project_dir / filename
    dest.write_text(text, encoding="utf-8")
    return dest
