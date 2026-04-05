"""PaddleOCR 3.x engine adapter."""
from __future__ import annotations

import logging
import os
import site
import sys
from pathlib import Path
from typing import Any, Optional

from app.orchestrator.services.ocr.base import OCREngine
from app.shared.logger import get_logger

logger = get_logger("ocr.paddleocr")

_OCR_VENV_PATH = Path(os.environ.get("OCR_VENV_PATH", "/opt/venvs/ocr"))
_PADDLE_WHEEL_INDEX = os.environ.get("PADDLE_WHEEL_INDEX", "cu126")
_BASE_IMAGE_CUDA = os.environ.get("BASE_IMAGE_CUDA", "12.8")


def _ensure_ocr_venv_site_packages() -> None:
    """Inject OCR venv site-packages into current interpreter import path."""
    if not _OCR_VENV_PATH.exists():
        return
    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_dir = _OCR_VENV_PATH / "lib" / py_ver / "site-packages"
    if not site_dir.exists():
        return
    site_path = str(site_dir)
    if site_path not in sys.path:
        site.addsitedir(site_path)


_ensure_ocr_venv_site_packages()


class _WarnOnce:
    """Emit a logger.warning exactly once per unique message per process."""
    _seen: set = set()

    @classmethod
    def warn(cls, log, msg: str) -> None:
        if msg not in cls._seen:
            cls._seen.add(msg)
            log.warning(msg)


def _get_paddleocr_version() -> str:
    """Return installed PaddleOCR version string, or 'unknown'."""
    try:
        import paddleocr
        return getattr(paddleocr, "__version__", "unknown")
    except Exception:
        return "unknown"


def _suppress_paddle_logs() -> None:
    """Suppress noisy ppocr / paddle log output regardless of PaddleOCR version."""
    for name in ("ppocr", "ppocr.utils", "ppocr.data", "paddle", "paddle.fluid"):
        logging.getLogger(name).setLevel(logging.ERROR)


class PaddleOCREngine(OCREngine):
    engine_id = "paddleocr"
    display_name = "PaddleOCR"
    description = "汎用高精度OCR。日本語・中国語・英語に対応。GPU推奨。"

    _ocr_instance = None
    # Caches the first initialization error so we fail fast on every subsequent
    # page instead of repeating the same exception message N times.
    _init_error: Optional[str] = None

    # GPU / device configuration (used by the enhanced pipeline)
    _device: str = "gpu:0"
    _resolved_device: str = "gpu:0"
    _use_layout: bool = False
    _smoke_test_passed: bool = False
    _smoke_test_warning: str = ""
    _last_error: Optional[str] = None
    _runtime_reason: str = ""

    @classmethod
    def configure(cls, device: str = "gpu:0", use_layout: bool = False) -> None:
        """Configure GPU device and layout mode for the enhanced pipeline.

        Call this before the first OCR invocation to set device preferences.
        If the instance is already created, it will be recreated on next init.
        """
        if cls._device != device or cls._use_layout != use_layout:
            cls._device = device
            cls._use_layout = use_layout
            # Force re-init on next call if device changed
            if cls._ocr_instance is not None:
                cls._ocr_instance = None
                cls._init_error = None
                cls._smoke_test_passed = False
                cls._smoke_test_warning = ""
                cls._last_error = None
                logger.info(f"PaddleOCR config changed: device={device} layout={use_layout}")

    @classmethod
    def run_paddle_ocr(cls, image_path: str, lang: str = "japan") -> list[dict[str, Any]]:
        """PaddleOCR 3.x 対応の単一呼び出し口."""
        cls._init_ocr(lang)
        if cls._ocr_instance is None:
            return []
        result = cls._ocr_instance.predict(str(image_path))
        normalized = cls._normalize_predict_result(result)
        return normalized

    def is_available(self) -> tuple[bool, str]:
        missing = []
        version = "unknown"
        try:
            import paddleocr  # noqa: F401
            version = getattr(paddleocr, "__version__", "unknown")
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

        diag = self._get_paddle_diag()
        if not diag["compiled_with_cuda"]:
            cls_reason = "paddlepaddle-gpu not active"
            PaddleOCREngine._runtime_reason = cls_reason
            return False, cls_reason

        # If init already failed in this process, report the cached error.
        if PaddleOCREngine._init_error is not None:
            return False, (
                f"PaddleOCR initialization failed (version={version}): "
                f"{PaddleOCREngine._init_error}"
            )

        return True, f"version={version}"

    def get_preprocessing_strategies(self) -> list[str]:
        return ["original", "grayscale"]

    def get_dependencies_info(self) -> dict:
        version = _get_paddleocr_version()
        return {
            "python_packages": [
                f"paddleocr (installed: {version})",
                "paddlepaddle-gpu>=2.5.0 (or paddlepaddle)",
                "Pillow>=10.0.0",
            ],
            "system_packages": [],
        }

    @staticmethod
    def _paddleocr_major_version() -> int:
        version = _get_paddleocr_version()
        try:
            return int(str(version).split(".")[0])
        except Exception:
            return 0

    @classmethod
    def _build_init_kwargs(cls, paddle_lang: str, force_no_layout: bool = False) -> dict:
        """Build PaddleOCR constructor kwargs for 3.x.

        ``force_no_layout`` disables the document layout kwargs even when
        ``_use_layout`` is True.  Used by the layout-retry fallback in
        ``_init_ocr`` when PaddleX raises a compatibility error.
        """
        kwargs: dict[str, Any] = {
            "lang": paddle_lang,
            "device": cls._resolved_device,
            "enable_hpi": False,
        }
        # 3.x の正式設定キーのみを利用（未対応環境でも安全に無視されるよう最小限）
        if cls._use_layout and not force_no_layout:
            kwargs["use_doc_orientation_classify"] = True
            kwargs["use_doc_unwarping"] = True
            kwargs["use_textline_orientation"] = True
        return kwargs

    @classmethod
    def _init_ocr(cls, paddle_lang: str) -> None:
        """Initialize singleton PaddleOCR instance with smoke test and layout retry.

        - PaddleOCR 3.x のみを許可
        - Caches initialization error in _init_error so calls fail fast (no per-page spam)
        - Stage 1: Constructor call; if layout kwargs trigger a compatibility error
          (e.g. "unexpected keyword argument 'cls'" from PaddleX internals), retries
          without layout kwargs.
        - Stage 2: Predict-time smoke test with a tiny dummy image; catches
          TypeError/RuntimeError that only surface at predict() time (not at __init__).
          Same layout-retry logic applies here.
        """
        if cls._ocr_instance is not None:
            return
        if cls._init_error is not None:
            raise RuntimeError(cls._init_error)
        cls._smoke_test_passed = False
        cls._smoke_test_warning = ""
        cls._last_error = None
        cls._runtime_reason = ""

        paddle_diag = cls._get_paddle_diag()
        if not paddle_diag["compiled_with_cuda"]:
            cls._init_error = "paddlepaddle-gpu not active"
            cls._last_error = cls._init_error
            cls._runtime_reason = cls._init_error
            raise RuntimeError(cls._init_error)

        from paddleocr import PaddleOCR

        _suppress_paddle_logs()

        version = _get_paddleocr_version()
        major = cls._paddleocr_major_version()
        if major != 3:
            cls._init_error = (
                f"Unsupported PaddleOCR major version: {version}. "
                "This service requires PaddleOCR 3.x."
            )
            cls._last_error = cls._init_error
            raise RuntimeError(cls._init_error)

        cls._resolved_device = cls._resolve_device(cls._device)
        logger.info(f"Initializing PaddleOCR (version={version}, lang={paddle_lang})")

        def _is_compat_error(exc: Exception) -> bool:
            """Return True for known PaddleX/Paddle version-mismatch errors."""
            msg = str(exc).lower()
            return "cls" in msg or "convertpirattribute" in msg

        # ── Stage 1: constructor (with layout-retry fallback) ────────────────
        kwargs = cls._build_init_kwargs(paddle_lang)
        try:
            instance = PaddleOCR(**kwargs)
        except Exception as exc:
            if cls._use_layout and _is_compat_error(exc):
                _WarnOnce.warn(
                    logger,
                    f"PaddleOCR layout init failed ({exc}); retrying without layout kwargs.",
                )
                try:
                    instance = PaddleOCR(**cls._build_init_kwargs(paddle_lang, force_no_layout=True))
                    logger.info("PaddleOCR initialized without layout kwargs (compatibility fallback).")
                except Exception as exc2:
                    cls._init_error = str(exc2)
                    cls._last_error = cls._init_error
                    logger.error(f"PaddleOCR initialization failed (version={version}): {exc2}")
                    raise RuntimeError(cls._init_error) from exc2
            else:
                cls._init_error = str(exc)
                cls._last_error = cls._init_error
                logger.error(f"PaddleOCR initialization failed (version={version}): {exc}")
                raise RuntimeError(cls._init_error) from exc

        # ── Stage 2: predict-time smoke test ────────────────────────────────
        # Catches errors that only appear when predict() is called (e.g. the
        # "unexpected keyword argument 'cls'" raised inside PaddleX pipeline).
        try:
            passed, warn = cls._run_smoke_test(instance)
            cls._smoke_test_passed = passed
            cls._smoke_test_warning = warn
            if passed:
                logger.info(f"PaddleOCR smoke test passed (version={version})")
            else:
                logger.warning(f"PaddleOCR smoke test warning (version={version}): {warn}")
        except TypeError as exc:
            if _is_compat_error(exc):
                _WarnOnce.warn(
                    logger,
                    f"PaddleOCR smoke test TypeError '{exc}'; disabling layout and reinitialising.",
                )
                try:
                    instance = PaddleOCR(**cls._build_init_kwargs(paddle_lang, force_no_layout=True))
                    passed, warn = cls._run_smoke_test(instance)
                    cls._smoke_test_passed = passed
                    cls._smoke_test_warning = warn
                    if passed:
                        logger.info("PaddleOCR smoke test passed after layout disable.")
                    else:
                        logger.warning(f"PaddleOCR smoke test warning after layout disable: {warn}")
                except Exception as exc3:
                    cls._init_error = str(exc3)
                    cls._last_error = cls._init_error
                    logger.error(f"PaddleOCR smoke test failed after retry: {exc3}")
                    raise RuntimeError(cls._init_error) from exc3
            else:
                cls._init_error = str(exc)
                cls._last_error = cls._init_error
                logger.error(f"PaddleOCR smoke test unexpected TypeError: {exc}")
                raise RuntimeError(cls._init_error) from exc
        except Exception as exc:
            cls._init_error = str(exc)
            cls._last_error = cls._init_error
            logger.error(f"PaddleOCR smoke test failed: {exc}")
            raise RuntimeError(cls._init_error) from exc

        # Assign only after both constructor and smoke test succeed.
        cls._ocr_instance = instance
        logger.info(f"PaddleOCR ready (version={version})")

    @staticmethod
    def _run_smoke_test(instance: Any) -> tuple[bool, str]:
        """Run minimum viable OCR on a synthetic image.

        Returns:
          (True, "") when at least one text region is recognized.
          (False, warning) when OCR call succeeds but returns no text.
        Raises:
          Any exception from predict() for fatal runtime failures.
        """
        import numpy as np
        from PIL import Image, ImageDraw

        image = Image.new("RGB", (192, 64), "white")
        draw = ImageDraw.Draw(image)
        draw.text((8, 18), "TEST", fill="black")
        arr = np.array(image)
        normalized = PaddleOCREngine._normalize_predict_result(instance.predict(arr))
        text_hits = [str(x.get("text", "")).strip() for x in normalized if str(x.get("text", "")).strip()]
        if text_hits:
            return True, ""
        return False, "OCR call succeeded but no text was detected in smoke image"

    @classmethod
    def get_runtime_status(cls) -> dict:
        """Detailed runtime state for startup log and status API."""
        version = _get_paddleocr_version()
        paddle_version = "unknown"
        paddlex_version = "unknown"
        cuda_compiled = False
        device = "unknown"
        try:
            import paddle
            paddle_version = getattr(paddle, "__version__", "unknown")
            cuda_compiled = bool(getattr(paddle, "is_compiled_with_cuda", lambda: False)())
            if hasattr(paddle, "device") and hasattr(paddle.device, "get_device"):
                device = paddle.device.get_device()
        except Exception:
            pass
        try:
            import paddlex
            paddlex_version = getattr(paddlex, "__version__", "unknown")
        except Exception:
            pass
        basic_ocr = bool(cls._smoke_test_passed and cls._ocr_instance is not None)
        return {
            "paddleocr_version": version,
            "paddlepaddle_version": paddle_version,
            "paddlex_version": paddlex_version,
            "cuda_compiled": cuda_compiled,
            "compiled_with_cuda": cuda_compiled,
            "device": device,
            "available": cls._init_error is None and cuda_compiled,
            "degraded": not cuda_compiled,
            "reason": cls._runtime_reason or ("paddlepaddle-gpu not active" if not cuda_compiled else ""),
            "configured_device": cls._device,
            "resolved_device": cls._resolved_device,
            "initialized": cls._ocr_instance is not None,
            "smoke_test_passed": cls._smoke_test_passed,
            "smoke_test_warning": cls._smoke_test_warning,
            "last_error": cls._last_error or cls._init_error,
            "base_image_cuda": _BASE_IMAGE_CUDA,
            "paddle_wheel_index": _PADDLE_WHEEL_INDEX,
            "basic_ocr": basic_ocr,
        }

    @staticmethod
    def _resolve_device(requested_device: str) -> str:
        """Resolve requested device to an actually usable Paddle device string."""
        req = (requested_device or "").strip().lower()
        if not req:
            req = "cpu"
        if req.startswith("cpu"):
            return "cpu"

        # GPU requested: use only when Paddle is CUDA-enabled.
        try:
            import paddle

            cuda_ok = bool(getattr(paddle, "is_compiled_with_cuda", lambda: False)())
            if not cuda_ok:
                raise RuntimeError("paddlepaddle-gpu not active")
            return requested_device
        except Exception as exc:
            raise RuntimeError(
                f"Could not activate requested Paddle device '{requested_device}': {exc}"
            ) from exc

    @staticmethod
    def _get_paddle_diag() -> dict[str, Any]:
        diag = {
            "paddle_version": "unknown",
            "compiled_with_cuda": False,
            "device": "unknown",
        }
        try:
            import paddle

            diag["paddle_version"] = getattr(paddle, "__version__", "unknown")
            diag["compiled_with_cuda"] = bool(paddle.is_compiled_with_cuda())
            diag["device"] = str(paddle.device.get_device())
        except Exception:
            pass
        return diag

    @staticmethod
    def _normalize_predict_result(result: Any) -> list[dict[str, Any]]:
        """Normalize PaddleOCR 3.x predict() output."""
        if not result:
            return []
        if not isinstance(result, list):
            result = [result]

        normalized: list[dict[str, Any]] = []
        for item in result:
            if hasattr(item, "res"):
                item = item.res
            if not isinstance(item, dict):
                continue

            texts = item.get("rec_texts") or []
            scores = item.get("rec_scores") or []
            polys = item.get("rec_polys") or item.get("dt_polys") or []
            for idx, text in enumerate(texts):
                poly = polys[idx] if idx < len(polys) else None
                score = scores[idx] if idx < len(scores) else 0.0
                normalized.append({
                    "text": str(text),
                    "score": float(score) if score is not None else 0.0,
                    "poly": poly,
                })

        return normalized

    def extract_text(
        self,
        image_path: Path,
        lang: str = "jpn+eng",
        preprocessing: Optional[str] = None,
    ) -> tuple[str, list[str]]:
        warnings: list[str] = []
        try:
            from paddleocr import PaddleOCR  # noqa: F401
        except ImportError:
            return "", ["paddleocr not installed – pip install paddleocr paddlepaddle-gpu"]

        # Fast-fail: if a previous initialization attempt failed, do not retry
        # for every single page – return the engine-level error once and stop.
        if PaddleOCREngine._init_error is not None:
            version = _get_paddleocr_version()
            return "", [
                f"PaddleOCR initialization failed (engine-level error, not retrying): "
                f"{PaddleOCREngine._init_error}",
                f"Installed PaddleOCR version: {version}",
            ]

        paddle_lang = "japan" if "jpn" in lang else "en"

        try:
            self._init_ocr(paddle_lang)

            result = self.run_paddle_ocr(str(image_path), paddle_lang)
            if not result:
                warnings.append(f"PaddleOCR returned empty result: {image_path.name}")
                return "", warnings

            lines = []
            for line_info in result:
                text = str(line_info.get("text", "")).strip()
                if text:
                    lines.append(text)

            combined = "\n".join(lines).strip()
            if not combined:
                warnings.append(f"PaddleOCR: no text extracted from {image_path.name}")
            else:
                logger.info(f"PaddleOCR ok: {image_path.name} → {len(combined)} chars")

            wrapped = f"===== OCR: {image_path.name} =====\n{combined}" if combined else ""
            return wrapped, warnings

        except RuntimeError as exc:
            # Propagated from _init_ocr when _init_error is set
            version = _get_paddleocr_version()
            return "", [
                f"PaddleOCR initialization failed: {exc}",
                f"Installed PaddleOCR version: {version}",
            ]
        except Exception as exc:
            # Unexpected per-page error (not an init failure)
            warnings.append(f"PaddleOCR failed: {image_path.name}: {exc}")
            logger.warning(f"PaddleOCR error for {image_path.name}: {exc}")
            return "", warnings
