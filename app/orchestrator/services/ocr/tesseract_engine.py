"""Tesseract OCR engine adapter."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from app.orchestrator.services.ocr.base import OCREngine
from app.orchestrator.services.ocr.numpy_safety import normalize_image_for_ocr
from app.shared.logger import get_logger

logger = get_logger("ocr.tesseract")


class TesseractEngine(OCREngine):
    engine_id = "tesseract"
    display_name = "Tesseract"
    description = "軽量・導入容易。apt install tesseract-ocr で利用可能。"

    def is_available(self) -> tuple[bool, str]:
        missing = []
        try:
            import pytesseract  # noqa: F401
        except ImportError:
            missing.append("pytesseract (pip install pytesseract)")

        try:
            from PIL import Image  # noqa: F401
        except ImportError:
            missing.append("Pillow (pip install pillow)")

        if not missing:
            # Check tesseract binary
            try:
                import pytesseract
                pytesseract.get_tesseract_version()
            except Exception:
                missing.append("tesseract-ocr バイナリ (apt install tesseract-ocr tesseract-ocr-jpn)")

        if missing:
            return False, ", ".join(missing)
        return True, ""

    def get_preprocessing_strategies(self) -> list[str]:
        return ["grayscale", "original", "binarize_soft", "binarize_hard", "invert"]

    def get_dependencies_info(self) -> dict:
        return {
            "python_packages": ["pytesseract>=0.3.10", "Pillow>=10.0.0"],
            "system_packages": ["tesseract-ocr", "tesseract-ocr-jpn", "tesseract-ocr-jpn-vert", "tesseract-ocr-eng"],
        }

    def extract_text(
        self,
        image_path: Path,
        lang: str = "jpn+eng",
        preprocessing: Optional[str] = None,
    ) -> tuple[str, list[str]]:
        warnings: list[str] = []
        try:
            from PIL import Image
        except ImportError:
            return "", ["Pillow not installed – pip install pillow"]
        try:
            import pytesseract  # noqa: F401
        except ImportError:
            return "", ["pytesseract not installed – pip install pytesseract"]

        try:
            image = Image.open(image_path)
            import numpy as np
            arr = np.asarray(image)
            pre_min = float(np.nanmin(arr)) if arr.size else 0.0
            pre_max = float(np.nanmax(arr)) if arr.size else 0.0
            normalized = normalize_image_for_ocr(arr)
            logger.debug(
                "Tesseract pre-normalize: dtype=%s shape=%s min=%.3f max=%.3f -> dtype=%s shape=%s",
                arr.dtype,
                tuple(arr.shape),
                pre_min,
                pre_max,
                normalized.dtype,
                tuple(normalized.shape),
            )
            image = Image.fromarray(normalized)
            # Upscale small images for better accuracy
            min_dim = 1000
            w, h = image.size
            if max(w, h) < min_dim:
                scale = min_dim / max(w, h)
                image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        except Exception as exc:
            return "", [f"Image open failed: {image_path.name}: {exc}"]

        strategies = [preprocessing] if preprocessing else self.get_preprocessing_strategies()
        best_text = ""

        for strategy in strategies:
            try:
                candidate = self._ocr_with_strategy(image, lang, strategy)
            except Exception as exc:
                warnings.append(f"OCR strategy '{strategy}' failed: {image_path.name}: {exc}")
                continue
            if len(candidate) > len(best_text):
                best_text = candidate
            if len(best_text) >= 20:
                break

        if not best_text:
            warnings.append(f"OCR returned empty result: {image_path.name}")
            logger.warning(f"OCR empty for {image_path.name} (tried {lang})")
        elif len(best_text) < 8:
            warnings.append(f"OCR near-empty result ({len(best_text)} chars): {image_path.name}")
            logger.warning(f"OCR near-empty for {image_path.name}: {best_text!r}")
        else:
            logger.info(f"OCR ok: {image_path.name} → {len(best_text)} chars")

        wrapped = f"===== OCR: {image_path.name} =====\n{best_text}" if best_text else ""
        return wrapped, warnings

    @staticmethod
    def _ocr_with_strategy(image: "Image.Image", lang: str, strategy: str) -> str:
        """Apply a preprocessing strategy then run tesseract."""
        import pytesseract

        if strategy == "original":
            proc = image.convert("RGB")
        elif strategy == "grayscale":
            proc = image.convert("L")
        elif strategy == "binarize_soft":
            proc = image.convert("L")
            proc = proc.point(lambda v: 255 if v > 128 else 0)
        elif strategy == "binarize_hard":
            proc = image.convert("L")
            proc = proc.point(lambda v: 255 if v > 160 else 0)
        elif strategy == "invert":
            from PIL import ImageOps
            proc = ImageOps.invert(image.convert("L"))
        elif strategy == "denoise":
            from PIL import ImageFilter
            proc = image.convert("L").filter(ImageFilter.MedianFilter(3))
        elif strategy == "adaptive_threshold":
            proc = image.convert("L")
            proc = proc.point(lambda v: 255 if v > 140 else 0)
        else:
            proc = image.convert("RGB")

        for attempt_lang in ([f"{lang}+jpn_vert", lang] if "jpn" in lang and "vert" not in lang else [lang]):
            try:
                text = pytesseract.image_to_string(proc, lang=attempt_lang, config="--psm 3")
                text = text.strip()
                if text:
                    return text
            except Exception:
                continue
        return ""
