"""PaddleOCR engine adapter."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from app.orchestrator.services.ocr.base import OCREngine
from app.shared.logger import get_logger

logger = get_logger("ocr.paddleocr")


class PaddleOCREngine(OCREngine):
    engine_id = "paddleocr"
    display_name = "PaddleOCR"
    description = "汎用高精度OCR。日本語・中国語・英語に対応。GPU推奨。"

    _ocr_instance = None

    def is_available(self) -> tuple[bool, str]:
        missing = []
        try:
            import paddleocr  # noqa: F401
        except ImportError:
            missing.append("paddleocr (pip install paddleocr)")
        try:
            import paddle  # noqa: F401
        except ImportError:
            missing.append("paddlepaddle (pip install paddlepaddle-gpu または paddlepaddle)")
        try:
            from PIL import Image  # noqa: F401
        except ImportError:
            missing.append("Pillow (pip install pillow)")

        if missing:
            return False, ", ".join(missing)
        return True, ""

    def get_preprocessing_strategies(self) -> list[str]:
        return ["original", "grayscale"]

    def get_dependencies_info(self) -> dict:
        return {
            "python_packages": ["paddleocr>=2.7.0", "paddlepaddle-gpu>=2.5.0 (or paddlepaddle)", "Pillow>=10.0.0"],
            "system_packages": [],
        }

    def extract_text(
        self,
        image_path: Path,
        lang: str = "jpn+eng",
        preprocessing: Optional[str] = None,
    ) -> tuple[str, list[str]]:
        warnings: list[str] = []
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            return "", ["paddleocr not installed – pip install paddleocr paddlepaddle-gpu"]

        paddle_lang = "japan" if "jpn" in lang else "en"

        try:
            if PaddleOCREngine._ocr_instance is None:
                PaddleOCREngine._ocr_instance = PaddleOCR(
                    use_angle_cls=True,
                    lang=paddle_lang,
                    show_log=False,
                )
            ocr = PaddleOCREngine._ocr_instance

            result = ocr.ocr(str(image_path), cls=True)
            if not result or not result[0]:
                warnings.append(f"PaddleOCR returned empty result: {image_path.name}")
                return "", warnings

            lines = []
            for line_info in result[0]:
                if line_info and len(line_info) >= 2:
                    text = line_info[1][0] if isinstance(line_info[1], (list, tuple)) else str(line_info[1])
                    lines.append(text)

            combined = "\n".join(lines).strip()
            if not combined:
                warnings.append(f"PaddleOCR: no text extracted from {image_path.name}")
            else:
                logger.info(f"PaddleOCR ok: {image_path.name} → {len(combined)} chars")

            wrapped = f"===== OCR: {image_path.name} =====\n{combined}" if combined else ""
            return wrapped, warnings
        except Exception as exc:
            warnings.append(f"PaddleOCR failed: {image_path.name}: {exc}")
            logger.warning(f"PaddleOCR error for {image_path.name}: {exc}")
            return "", warnings
