"""NDLOCR-Lite engine adapter for Japanese document OCR.

This adapter uses a vendored upstream checkout at /opt/ndlocr-lite that is
prepared during Docker image build (fixed commit checkout).

Availability criteria (all must pass):
  1. /opt/ndlocr-lite/src/ocr.py exists
  2. /opt/ndlocr-lite/src/model/*.onnx has exactly 4 files
  3. /opt/ndlocr-lite/src/config/ndl.yaml and NDLmoji.yaml exist
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Optional

from app.orchestrator.services.ocr.base import OCREngine
from app.shared.logger import get_logger

logger = get_logger("ocr.ndlocr_lite")

_NDLOCR_ROOT = Path("/opt/ndlocr-lite")
_OCR_SCRIPT = _NDLOCR_ROOT / "src" / "ocr.py"
_MODEL_DIR = _NDLOCR_ROOT / "src" / "model"
_CONFIG_DIR = _NDLOCR_ROOT / "src" / "config"
_REQUIRED_CONFIGS = ("ndl.yaml", "NDLmoji.yaml")
_NDLOCR_VENV = Path(os.environ.get("NDLOCR_VENV_PATH", "/opt/venvs/ndlocr"))
_NDLOCR_PYTHON = Path(os.environ.get("NDLOCR_PYTHON", str(_NDLOCR_VENV / "bin" / "python")))
_NDLOCR_DEVICE = os.environ.get("NDLOCR_DEVICE", "cpu").strip().lower()


def _python_binary() -> Optional[str]:
    if _NDLOCR_PYTHON.is_file():
        return str(_NDLOCR_PYTHON)
    return None


def _requested_device() -> str:
    return "cuda" if _NDLOCR_DEVICE == "cuda" else "cpu"


@lru_cache(maxsize=1)
def _detect_onnxruntime_backend(py: str) -> str:
    cmd = [
        py,
        "-c",
        (
            "import onnxruntime as ort; "
            "providers=ort.get_available_providers(); "
            "print('cuda' if 'CUDAExecutionProvider' in providers else 'cpu')"
        ),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=20,
        )
    except Exception:
        return "unknown"
    if result.returncode != 0:
        return "missing"
    out = (result.stdout or "").strip().lower()
    if out in ("cpu", "cuda"):
        return out
    return "unknown"


def _check_ocr_script() -> tuple[bool, str]:
    if _OCR_SCRIPT.is_file():
        return True, ""
    return False, f"NDLOCR-Lite OCR script not found: {_OCR_SCRIPT}"


def _check_model_and_config_files() -> tuple[bool, str, dict]:
    model_files = sorted(_MODEL_DIR.glob("*.onnx")) if _MODEL_DIR.is_dir() else []
    configs_present = {
        name: (_CONFIG_DIR / name).is_file()
        for name in _REQUIRED_CONFIGS
    }

    details = {
        "model_dir_exists": _MODEL_DIR.is_dir(),
        "model_files_count": len(model_files),
        "config_dir_exists": _CONFIG_DIR.is_dir(),
        "configs_present": configs_present,
        "status": "ready",
    }

    if len(model_files) != 4:
        details["status"] = "incomplete"
        return (
            False,
            f"NDLOCR-Lite model files mismatch: expected 4 onnx, found {len(model_files)} ({_MODEL_DIR})",
            details,
        )

    missing_configs = [name for name, ok in configs_present.items() if not ok]
    if missing_configs:
        details["status"] = "incomplete"
        return (
            False,
            f"NDLOCR-Lite config files missing in {_CONFIG_DIR}: {', '.join(missing_configs)}",
            details,
        )

    return True, "", details


def get_ndlocr_status() -> dict:
    """Return a detailed status dict for the NDLOCR-Lite engine."""
    py = _python_binary()
    requested_device = _requested_device()
    script_ok, script_err = _check_ocr_script()
    assets_ok, assets_err, asset_details = _check_model_and_config_files()

    errors = []
    if py is None:
        errors.append(f"NDLOCR-Lite venv python not found: {_NDLOCR_PYTHON}")
    if not script_ok:
        errors.append(script_err)
    if not assets_ok:
        errors.append(assets_err)

    backend = _detect_onnxruntime_backend(py) if py is not None else "missing"
    if backend == "missing":
        errors.append("NDLOCR-Lite unavailable: missing dependency onnxruntime")
    if requested_device == "cuda" and backend != "cuda":
        errors.append(
            "NDLOCR-Lite requested cuda but CUDAExecutionProvider is unavailable"
        )

    runtime_ready = py is not None and script_ok and assets_ok and not errors

    return {
        "status": "ready" if runtime_ready else "unavailable",
        "venv": str(_NDLOCR_VENV),
        "python": str(_NDLOCR_PYTHON),
        "device_request": requested_device,
        "onnxruntime_backend": backend,
        "reason": "  |  ".join(errors) if errors else "",
        "installed": script_ok,
        "importable": script_ok,
        "cli_functional": script_ok,
        "model_files_present": assets_ok,
        "runtime_ready": runtime_ready,
        "model_path": str(_MODEL_DIR),
        "model_dir_status": asset_details["status"],
        "model_dir_exists": asset_details["model_dir_exists"],
        "model_files_count": asset_details["model_files_count"],
        "aux_files_count": sum(asset_details["configs_present"].values()),
        "ocr_script_path": str(_OCR_SCRIPT),
        "ocr_script_exists": script_ok,
        "config_dir_exists": asset_details["config_dir_exists"],
        "configs_present": asset_details["configs_present"],
        "error_message": "  |  ".join(errors) if errors else "",
    }


class NDLOCRLiteEngine(OCREngine):
    engine_id = "ndlocr_lite"
    display_name = "NDLOCR-Lite"
    description = "国立国会図書館開発の日本語文書向けOCR。縦書き・歴史的文書に強い。"

    def is_available(self) -> tuple[bool, str]:
        status = get_ndlocr_status()
        if status["runtime_ready"]:
            return True, ""
        return False, status["error_message"]

    def get_preprocessing_strategies(self) -> list[str]:
        return ["original", "grayscale"]

    def get_dependencies_info(self) -> dict:
        status = get_ndlocr_status()
        return {
            "python_packages": ["Pillow>=10.0.0", "onnxruntime==1.23.2"],
            "system_packages": [],
            "notes": (
                "NDLOCR-Lite は Docker build 時に /opt/ndlocr-lite へ固定コミットで配置され、"
                "実行時は専用 venv の python で /opt/ndlocr-lite/src/ocr.py を実行します。"
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

        status = get_ndlocr_status()
        if not status["runtime_ready"]:
            warnings.append(f"NDLOCR-Lite は利用不可: {status['error_message']}")
            return "", warnings

        py = _python_binary()
        if py is None:
            warnings.append(f"NDLOCR-Lite unavailable: missing venv python {_NDLOCR_PYTHON}")
            return "", warnings

        with tempfile.TemporaryDirectory(prefix="ndlocr_") as tmpdir:
            tmp = Path(tmpdir)
            input_dir = tmp / "input"
            output_dir = tmp / "output"
            input_dir.mkdir()
            output_dir.mkdir()

            dest = input_dir / image_path.name
            shutil.copy2(image_path, dest)

            cmd = [
                py,
                str(_OCR_SCRIPT),
                "-i", str(input_dir),
                "-o", str(output_dir),
                "--model_path", str(_MODEL_DIR),
                "--device", status["device_request"],
            ]
            logger.debug(f"NDLOCR-Lite cmd: {' '.join(cmd)}")

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=180,
                    cwd=str(_NDLOCR_ROOT),
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
                missing_onnx = "No module named 'onnxruntime'" in (result.stderr or "")
                warnings.append(
                    "NDLOCR-Lite unavailable: missing dependency onnxruntime"
                    if missing_onnx
                    else (
                        f"NDLOCR-Lite failed (rc={result.returncode}): "
                        f"{image_path.name}: {stderr_snippet}"
                    )
                )
                logger.warning(f"NDLOCR-Lite rc={result.returncode} for {image_path.name}")
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
