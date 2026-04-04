"""OCR engine abstraction layer and enhanced pipeline."""
from app.orchestrator.services.ocr.base import OCREngine
from app.orchestrator.services.ocr.factory import get_engine, list_engines

__all__ = ["OCREngine", "get_engine", "list_engines"]

# Lazy imports for the enhanced pipeline – avoids circular imports at module load
def get_pipeline():
    """Return the run_pipeline function from the enhanced pipeline module."""
    from app.orchestrator.services.ocr.pipeline import run_pipeline
    return run_pipeline
