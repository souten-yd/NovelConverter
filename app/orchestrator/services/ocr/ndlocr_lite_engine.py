"""NDLOCR-Lite engine adapter for Japanese document OCR."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from app.orchestrator.services.ocr.base import OCREngine
from app.shared.logger import get_logger

logger = get_logger("ocr.ndlocr_lite")


class NDLOCRLiteEngine(OCREngine):
    engine_id = "ndlocr_lite"
    display_name = "NDLOCR-Lite"
    description = "国立国会図書館開発の日本語文書向けOCR。縦書き・歴史的文書に強い。"

    def is_available(self) -> tuple[bool, str]:
        missing = []
        try:
            import ndlocr_cli  # noqa: F401
        except ImportError:
            try:
                # Some installations use a different package name
                import ndlocr  # noqa: F401
            except ImportError:
                missing.append("ndlocr-cli (pip install ndlocr-cli または git clone + pip install)")
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
            "python_packages": ["ndlocr-cli", "Pillow>=10.0.0"],
            "system_packages": [],
            "notes": "NDLOCRはPyTorch + 専用モデルが必要。初回起動時にモデルがダウンロードされます。",
        }

    def extract_text(
        self,
        image_path: Path,
        lang: str = "jpn+eng",
        preprocessing: Optional[str] = None,
    ) -> tuple[str, list[str]]:
        warnings: list[str] = []

        try:
            # Try to use ndlocr_cli
            import subprocess
            result = subprocess.run(
                ["ndlocr", "infer", str(image_path)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                warnings.append(f"NDLOCR-Lite failed: {image_path.name}: {result.stderr[:200]}")
                return "", warnings

            text = result.stdout.strip()
            if not text:
                warnings.append(f"NDLOCR-Lite returned empty result: {image_path.name}")
            else:
                logger.info(f"NDLOCR-Lite ok: {image_path.name} → {len(text)} chars")

            wrapped = f"===== OCR: {image_path.name} =====\n{text}" if text else ""
            return wrapped, warnings
        except FileNotFoundError:
            warnings.append("NDLOCR-Lite: ndlocr コマンドが見つかりません。インストールを確認してください。")
            return "", warnings
        except Exception as exc:
            warnings.append(f"NDLOCR-Lite failed: {image_path.name}: {exc}")
            logger.warning(f"NDLOCR-Lite error for {image_path.name}: {exc}")
            return "", warnings
