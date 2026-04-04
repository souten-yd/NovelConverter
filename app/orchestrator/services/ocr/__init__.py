"""OCR engine abstraction layer."""
from app.orchestrator.services.ocr.base import OCREngine
from app.orchestrator.services.ocr.factory import get_engine, list_engines

__all__ = ["OCREngine", "get_engine", "list_engines"]
