"""OCR engine registry and factory."""
from __future__ import annotations

from typing import Optional

from app.orchestrator.services.ocr.base import OCREngine
from app.orchestrator.services.ocr.tesseract_engine import TesseractEngine
from app.orchestrator.services.ocr.paddleocr_engine import PaddleOCREngine
from app.orchestrator.services.ocr.qwen_vl_engine import QwenVLEngine
from app.orchestrator.services.ocr.ndlocr_lite_engine import NDLOCRLiteEngine

# Singleton engine instances
_engines: dict[str, OCREngine] = {
    "tesseract": TesseractEngine(),
    "paddleocr": PaddleOCREngine(),
    "qwen_vl": QwenVLEngine(),
    "ndlocr_lite": NDLOCRLiteEngine(),
}

# Engine aliases: the enhanced pipeline uses these names
_ENGINE_ALIASES: dict[str, str] = {
    "paddle_fast": "paddleocr",
    "paddle_layout": "paddleocr",
    "fallback": "tesseract",
}

DEFAULT_ENGINE = "tesseract"


def get_engine(engine_id: str) -> OCREngine:
    """Get an OCR engine by ID. Falls back to tesseract if not found.

    Also resolves enhanced-pipeline aliases:
      paddle_fast / paddle_layout → paddleocr
      fallback → tesseract
    """
    resolved = _ENGINE_ALIASES.get(engine_id, engine_id)
    engine = _engines.get(resolved)
    if engine is None:
        engine = _engines[DEFAULT_ENGINE]
    return engine


def list_engines() -> list[dict]:
    """List all registered engines with availability status."""
    result = []
    for eid, engine in _engines.items():
        available, missing_deps = engine.is_available()
        deps = engine.get_dependencies_info()
        result.append({
            "id": eid,
            "name": engine.display_name,
            "description": engine.description,
            "available": available,
            "missing_deps": missing_deps,
            "dependencies": deps,
        })
    return result


def get_available_engines() -> list[str]:
    """Return IDs of engines that are currently available."""
    return [eid for eid, engine in _engines.items() if engine.is_available()[0]]
