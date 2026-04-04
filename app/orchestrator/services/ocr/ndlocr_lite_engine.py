"""NDLOCR-Lite engine adapter for Japanese document OCR."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
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

        # Check ndlocr CLI binary (installed from GitHub)
        if not shutil.which("ndlocr"):
            missing.append("ndlocr コマンド (pip install git+https://github.com/ndl-lab/ndlocr_cli.git)")

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
            "python_packages": ["Pillow>=10.0.0"],
            "system_packages": [],
            "notes": (
                "NDLOCRはGitHubからインストール: "
                "pip install git+https://github.com/ndl-lab/ndlocr_cli.git  "
                "初回実行時にモデルが自動ダウンロードされます。"
            ),
        }

    def extract_text(
        self,
        image_path: Path,
        lang: str = "jpn+eng",
        preprocessing: Optional[str] = None,
    ) -> tuple[str, list[str]]:
        warnings: list[str] = []

        if not shutil.which("ndlocr"):
            warnings.append(
                "NDLOCR-Lite: ndlocr コマンドが見つかりません。"
                "pip install git+https://github.com/ndl-lab/ndlocr_cli.git"
            )
            return "", warnings

        # ndlocr v2 operates on a directory of images, not a single file.
        # Create a temporary input/output directory pair.
        with tempfile.TemporaryDirectory(prefix="ndlocr_") as tmpdir:
            tmp = Path(tmpdir)
            input_dir = tmp / "input"
            output_dir = tmp / "output"
            input_dir.mkdir()
            output_dir.mkdir()

            # Copy image into input dir (ndlocr expects images in the dir)
            dest = input_dir / image_path.name
            shutil.copy2(image_path, dest)

            try:
                result = subprocess.run(
                    [
                        "ndlocr",
                        "-i", str(input_dir),
                        "-o", str(output_dir),
                        "--use_gpu", "False",
                    ],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=180,
                )
            except subprocess.TimeoutExpired:
                warnings.append(f"NDLOCR-Lite timeout: {image_path.name}")
                logger.warning(f"NDLOCR-Lite timed out for {image_path.name}")
                return "", warnings
            except Exception as exc:
                warnings.append(f"NDLOCR-Lite subprocess error: {image_path.name}: {exc}")
                logger.warning(f"NDLOCR-Lite error for {image_path.name}: {exc}")
                return "", warnings

            if result.returncode != 0:
                stderr_snippet = result.stderr[:300] if result.stderr else "(no stderr)"
                warnings.append(f"NDLOCR-Lite failed (rc={result.returncode}): {image_path.name}: {stderr_snippet}")
                logger.warning(f"NDLOCR-Lite rc={result.returncode} for {image_path.name}")
                return "", warnings

            # Collect all text files written to output_dir
            texts: list[str] = []
            for txt_file in sorted(output_dir.rglob("*.txt")):
                try:
                    texts.append(txt_file.read_text(encoding="utf-8"))
                except Exception:
                    pass

            # Also check stdout
            if result.stdout and result.stdout.strip():
                texts.append(result.stdout.strip())

            combined = "\n".join(t for t in texts if t).strip()

        if not combined:
            warnings.append(f"NDLOCR-Lite returned empty result: {image_path.name}")
            logger.warning(f"NDLOCR-Lite empty result for {image_path.name}")
        else:
            logger.info(f"NDLOCR-Lite ok: {image_path.name} → {len(combined)} chars")

        wrapped = f"===== OCR: {image_path.name} =====\n{combined}" if combined else ""
        return wrapped, warnings
