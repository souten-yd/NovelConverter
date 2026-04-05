"""Hash-based OCR result cache.

Caches OCR results per-page to avoid redundant re-processing.
The cache key is derived from:
  - image file content hash
  - engine name
  - ruby mode
  - preprocessing options

Cache entries are stored as JSON files in a configurable directory.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from app.shared.logger import get_logger
from app.orchestrator.services.ocr.numpy_safety import safe_for_json

logger = get_logger("ocr.cache")


def _compute_cache_key(
    image_path: str,
    engine: str,
    ruby_mode: str = "none",
    extra: str = "",
) -> str:
    """Compute a stable cache key for an OCR job."""
    hasher = hashlib.sha256()

    # Hash image content
    try:
        with open(image_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
    except Exception:
        # If we can't read the file, include the path as fallback
        hasher.update(image_path.encode())

    hasher.update(engine.encode())
    hasher.update(ruby_mode.encode())
    if extra:
        hasher.update(extra.encode())

    return hasher.hexdigest()


class OCRCache:
    """File-system backed OCR result cache."""

    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir:
            self._dir = Path(cache_dir)
        else:
            self._dir = Path("/tmp/ocr_cache")
        self._dir.mkdir(parents=True, exist_ok=True)
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def get(
        self,
        image_path: str,
        engine: str,
        ruby_mode: str = "none",
    ) -> Optional[dict]:
        """Retrieve a cached result, or None if not found."""
        if not self._enabled:
            return None

        key = _compute_cache_key(image_path, engine, ruby_mode)
        cache_path = self._dir / f"{key}.json"
        if not cache_path.exists():
            return None

        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            logger.debug(f"Cache hit: {key[:12]}... for {Path(image_path).name}")
            return data
        except Exception as exc:
            logger.debug(f"Cache read error: {exc}")
            return None

    def put(
        self,
        image_path: str,
        engine: str,
        ruby_mode: str,
        result: dict,
    ) -> None:
        """Store an OCR result in the cache."""
        if not self._enabled:
            return

        key = _compute_cache_key(image_path, engine, ruby_mode)
        cache_path = self._dir / f"{key}.json"
        try:
            normalized = safe_for_json(result)
            cache_path.write_text(
                json.dumps(normalized, ensure_ascii=False, indent=None),
                encoding="utf-8",
            )
            logger.debug(f"Cache stored: {key[:12]}... for {Path(image_path).name}")
        except Exception as exc:
            logger.debug(f"Cache write error: {exc}")

    def clear(self) -> int:
        """Remove all cached results.  Returns count of removed entries."""
        count = 0
        for f in self._dir.glob("*.json"):
            try:
                f.unlink()
                count += 1
            except Exception:
                pass
        logger.info(f"Cache cleared: {count} entries removed")
        return count

    def size(self) -> int:
        """Return the number of cached entries."""
        return len(list(self._dir.glob("*.json")))
