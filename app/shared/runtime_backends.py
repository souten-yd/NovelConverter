"""Runtime backend auto-detection for local Windows AMD environments."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from app.shared.paths import get_data_dir, get_repo_root


_RUNTIME_ENV_KEY = "NOVELCONVERTER_RUNTIME_STATUS"


def _is_windows() -> bool:
    return sys.platform.startswith("win")


def _has_vulkan() -> bool:
    # Prefer explicit vulkan DLL presence check on Windows.
    if _is_windows():
        win_dir = Path(os.environ.get("WINDIR", r"C:\Windows"))
        if (win_dir / "System32" / "vulkan-1.dll").exists():
            return True
    return shutil.which("vulkaninfo") is not None


def _detect_llm_backend() -> str:
    # llama.cpp Vulkan is preferred when Vulkan runtime appears available.
    return "vulkan" if _has_vulkan() else "cpu"


def _detect_torch_backend() -> str:
    try:
        import torch_directml  # type: ignore

        dev = torch_directml.device()
        return "directml" if dev is not None else "cpu"
    except Exception:
        return "cpu"


def _detect_onnx_backend() -> str:
    try:
        import onnxruntime as ort  # type: ignore

        providers = ort.get_available_providers()
        return "directml" if "DmlExecutionProvider" in providers else "cpu"
    except Exception:
        return "cpu"


def _detect_ocr_backend() -> str:
    # Windows AMD non-ROCm default: PaddleOCR on CPU.
    try:
        from app.orchestrator.services.ocr.ndlocr_lite_engine import get_ndlocr_status

        ndl = get_ndlocr_status()
        if ndl.get("runtime_ready"):
            return "ndlocr"
    except Exception:
        pass

    # If Tesseract is missing we still advertise paddle_cpu as baseline path.
    if shutil.which("tesseract"):
        return "tesseract"
    return "paddle_cpu"


def detect_runtime_backends() -> dict[str, Any]:
    data_dir = get_data_dir()
    repo_root = get_repo_root()
    return {
        "platform": "windows" if _is_windows() else sys.platform,
        "selected_llm_backend": _detect_llm_backend(),
        "selected_torch_backend": _detect_torch_backend(),
        "selected_onnx_backend": _detect_onnx_backend(),
        "selected_ocr_backend": _detect_ocr_backend(),
        "venv_path": str(repo_root / ".venv"),
        "models_root": str(data_dir / "models"),
        "runtime_llama_root": str(repo_root / "runtime" / "llama"),
        "first_run": False,
        "reused_env": True,
    }


def save_runtime_status(status: dict[str, Any]) -> None:
    os.environ[_RUNTIME_ENV_KEY] = json.dumps(status, ensure_ascii=False)


def load_runtime_status() -> dict[str, Any]:
    raw = os.environ.get(_RUNTIME_ENV_KEY, "")
    if raw:
        try:
            return json.loads(raw)
        except Exception:
            pass
    return detect_runtime_backends()


def command_supports_vulkan(llama_bin: str) -> bool:
    """Best-effort check: verify llama-server accepts Vulkan-related flags."""
    try:
        cp = subprocess.run(
            [llama_bin, "--help"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=10,
        )
        text = f"{cp.stdout}\n{cp.stderr}".lower()
        return "vulkan" in text or "--gpu-layers" in text or "--n-gpu-layers" in text
    except Exception:
        return False
