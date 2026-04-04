"""Abstract base class for OCR engines."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class OCREngine(ABC):
    """Interface that all OCR engines must implement."""

    engine_id: str = ""
    display_name: str = ""
    description: str = ""

    @abstractmethod
    def is_available(self) -> tuple[bool, str]:
        """Check if this engine is ready to use.

        Returns (available, message).
        - (True, "") if ready
        - (False, "reason or missing deps") if not
        """
        ...

    @abstractmethod
    def extract_text(
        self,
        image_path: Path,
        lang: str = "jpn+eng",
        preprocessing: Optional[str] = None,
    ) -> tuple[str, list[str]]:
        """Run OCR on a single image file.

        Args:
            image_path: Path to the image file.
            lang: Language hint (engine-specific interpretation).
            preprocessing: Optional preprocessing strategy override.

        Returns:
            (extracted_text, warnings)
        """
        ...

    def get_preprocessing_strategies(self) -> list[str]:
        """Return list of supported preprocessing strategy names."""
        return []

    def get_dependencies_info(self) -> dict:
        """Return info about required dependencies."""
        return {
            "python_packages": [],
            "system_packages": [],
        }
