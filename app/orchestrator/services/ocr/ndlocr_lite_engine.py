"""NDLOCR-Lite engine adapter for Japanese document OCR.

NDLOCR-Lite is the National Diet Library's Japanese OCR engine.
It is strong on vertical Japanese text and historical documents.

Installation:
  pip install git+https://github.com/ndl-lab/ndlocr-lite.git

Model storage:
  Models are NOT included in the pip package and must be downloaded separately.
  By default they are stored at NDLOCR_MODEL_DIR (env) or /workspace/ndlocr_models.
  The entrypoint.sh startup script will attempt to download them automatically.

Availability criteria (all must pass):
  1. NDLOCR-Lite CLI binary is in PATH (ndlocr-lite preferred, ndlocr fallback)
  2. CLI --help runs without error (CLI is functional)
  3. NDLOCR_MODEL_DIR exists and contains at least one model file
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from app.orchestrator.services.ocr.base import OCREngine
from app.shared.logger import get_logger

logger = get_logger("ocr.ndlocr_lite")

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_DEFAULT_MODEL_DIR = "/workspace/ndlocr_models"

# File extensions that indicate a downloaded model file
_MODEL_EXTENSIONS = {".pth", ".pt", ".onnx", ".pdparams", ".bin", ".npz"}


def _get_model_dir() -> Path:
    """Return model directory (env NDLOCR_MODEL_DIR or default)."""
    return Path(os.environ.get("NDLOCR_MODEL_DIR", _DEFAULT_MODEL_DIR))


def _ndlocr_binary() -> Optional[str]:
    """Return NDLOCR-Lite CLI path (ndlocr-lite preferred, ndlocr fallback)."""
    return shutil.which("ndlocr-lite") or shutil.which("ndlocr")


def _check_cli_functional() -> tuple[bool, str]:
    """Verify NDLOCR-Lite CLI is installed and responds to --help.

    Returns (ok, error_message).
    """
    binary = _ndlocr_binary()
    if binary is None:
        return False, (
            "NDLOCR-Lite CLI が見つかりません。"
            "pip install git+https://github.com/ndl-lab/ndlocr-lite.git"
        )

    try:
        proc = subprocess.run(
            [binary, "--help"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=15,
        )
        # --help commonly exits 0 or 1; both are acceptable here.
        if proc.returncode not in (0, 1):
            stderr = (proc.stderr or "").strip()[:300]
            return False, (
                f"NDLOCR-Lite --help が異常終了しました (rc={proc.returncode}): {stderr}"
            )
        return True, ""
    except FileNotFoundError:
        return False, f"NDLOCR-Lite バイナリが見つかりません: {binary}"
    except subprocess.TimeoutExpired:
        return False, "NDLOCR-Lite --help がタイムアウトしました (CLI が壊れている可能性があります)"
    except Exception as exc:
        return False, f"NDLOCR-Lite CLI 確認中にエラーが発生しました: {exc}"


def _check_model_files() -> tuple[bool, str]:
    """Check that model files exist in the configured model directory.

    Returns (ok, error_message).
    """
    model_dir = _get_model_dir()

    if not model_dir.exists():
        return False, (
            f"モデルディレクトリが見つかりません: {model_dir}  "
            f"(環境変数 NDLOCR_MODEL_DIR={model_dir} に ndlocr モデルを配置してください。"
            "起動スクリプト entrypoint.sh で自動ダウンロードを試みます)"
        )

    model_files = [
        p for p in model_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in _MODEL_EXTENSIONS
    ]
    if not model_files:
        return False, (
            f"モデルファイルが見つかりません: {model_dir} にモデルが未配置です  "
            f"(対象拡張子: {', '.join(sorted(_MODEL_EXTENSIONS))})。"
            "entrypoint.sh が起動時に自動ダウンロードを試みます。"
            "手動で実行する場合は scripts/download_ndlocr_models.sh を参照してください。"
        )

    logger.debug(f"NDLOCR-Lite: {len(model_files)} model file(s) found in {model_dir}")
    return True, ""


def get_ndlocr_status() -> dict:
    """Return a detailed status dict for the NDLOCR-Lite engine.

    Used by health-check endpoints and startup diagnostics.
    Keys:
      installed        – ndlocr binary present in PATH
      importable       – (reserved; ndlocr is CLI-based, not imported)
      cli_functional   – ndlocr --help succeeds
      model_files_present – model files found in model_dir
      runtime_ready    – all of the above are True
      model_path       – configured model directory path
      error_message    – human-readable summary of what is missing
    """
    binary = _ndlocr_binary()
    installed = binary is not None
    cli_ok, cli_err = _check_cli_functional() if installed else (False, "ndlocr not found")
    model_ok, model_err = _check_model_files()
    model_dir = _get_model_dir()

    errors = []
    if not installed:
        errors.append(
            "NDLOCR-Lite CLI が PATH にありません "
            "(pip install git+https://github.com/ndl-lab/ndlocr-lite.git)"
        )
    elif not cli_ok:
        errors.append(f"CLI 動作確認失敗: {cli_err}")
    if not model_ok:
        errors.append(f"モデル未配置: {model_err}")

    runtime_ready = installed and cli_ok and model_ok

    return {
        "installed": installed,
        "importable": installed,  # CLI-based; same as installed
        "cli_functional": cli_ok,
        "model_files_present": model_ok,
        "runtime_ready": runtime_ready,
        "model_path": str(model_dir),
        "error_message": "  |  ".join(errors) if errors else "",
    }


# ---------------------------------------------------------------------------
# Engine adapter
# ---------------------------------------------------------------------------

class NDLOCRLiteEngine(OCREngine):
    engine_id = "ndlocr_lite"
    display_name = "NDLOCR-Lite"
    description = "国立国会図書館開発の日本語文書向けOCR。縦書き・歴史的文書に強い。"

    def is_available(self) -> tuple[bool, str]:
        """Return (available, error_message).

        Availability requires:
          1. ndlocr CLI in PATH
          2. ndlocr --help succeeds
          3. Model files present in NDLOCR_MODEL_DIR
        """
        status = get_ndlocr_status()
        if status["runtime_ready"]:
            return True, ""
        return False, status["error_message"]

    def get_preprocessing_strategies(self) -> list[str]:
        return ["original", "grayscale"]

    def get_dependencies_info(self) -> dict:
        status = get_ndlocr_status()
        return {
            "python_packages": ["Pillow>=10.0.0"],
            "system_packages": [],
            "notes": (
                "NDLOCR-Lite は GitHub からインストール: "
                "pip install git+https://github.com/ndl-lab/ndlocr-lite.git  "
                "モデルは初回起動時に NDLOCR_MODEL_DIR へ自動ダウンロードされます。"
            ),
            "status": status,
        }

    def extract_text(
        self,
        image_path: Path,
        lang: str = "jpn+eng",
        preprocessing: Optional[str] = None,
    ) -> tuple[str, list[str]]:
        warnings: list[str] = []

        # Pre-flight check before touching the file
        status = get_ndlocr_status()
        if not status["runtime_ready"]:
            warnings.append(
                f"NDLOCR-Lite は利用不可: {status['error_message']}"
            )
            return "", warnings

        binary = _ndlocr_binary()
        model_dir = _get_model_dir()

        # ndlocr v2 operates on a directory of images, not a single file.
        with tempfile.TemporaryDirectory(prefix="ndlocr_") as tmpdir:
            tmp = Path(tmpdir)
            input_dir = tmp / "input"
            output_dir = tmp / "output"
            input_dir.mkdir()
            output_dir.mkdir()

            dest = input_dir / image_path.name
            shutil.copy2(image_path, dest)

            cmd = [
                binary,
                "-i", str(input_dir),
                "-o", str(output_dir),
                "--use_gpu", "False",
                "--model_path", str(model_dir),
            ]
            logger.debug(f"NDLOCR-Lite cmd: {' '.join(cmd)}")

            try:
                result = subprocess.run(
                    cmd,
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
                stderr_snippet = (result.stderr or "")[:300] or "(no stderr)"
                warnings.append(
                    f"NDLOCR-Lite failed (rc={result.returncode}): "
                    f"{image_path.name}: {stderr_snippet}"
                )
                logger.warning(
                    f"NDLOCR-Lite rc={result.returncode} for {image_path.name}"
                )
                return "", warnings

            texts: list[str] = []
            for txt_file in sorted(output_dir.rglob("*.txt")):
                try:
                    texts.append(txt_file.read_text(encoding="utf-8"))
                except Exception:
                    pass

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
