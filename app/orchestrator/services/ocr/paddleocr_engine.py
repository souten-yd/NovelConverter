"""PaddleOCR engine adapter.

Version compatibility notes:
  - PaddleOCR 2.x: PaddleOCR(show_log=False, ...) is accepted
  - PaddleOCR 3.x: show_log was removed; suppress logging via Python's logging module
  This adapter detects the installed version and builds kwargs accordingly.
"""
from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Optional

from app.orchestrator.services.ocr.base import OCREngine
from app.shared.logger import get_logger

logger = get_logger("ocr.paddleocr")


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
    _use_layout: bool = False

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
                logger.info(f"PaddleOCR config changed: device={device} layout={use_layout}")

    @classmethod
    def get_raw_ocr_result(cls, image_path: str, lang: str = "japan") -> list:
        """Return raw PaddleOCR result with bounding boxes and confidence.

        Used by the enhanced pipeline for token-level extraction.
        Returns the raw result list from paddleocr.ocr().
        """
        paddle_lang = "japan" if "jpn" in lang or lang == "japan" else "en"
        cls._init_ocr(paddle_lang)
        if cls._ocr_instance is None:
            return []
        result = cls._ocr_instance.ocr(str(image_path), cls=True)
        return result if result else []

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

        # If init already failed in this process, report the cached error.
        if PaddleOCREngine._init_error is not None:
            return False, (
                f"PaddleOCR initialization failed (version={version}): "
                f"{PaddleOCREngine._init_error}"
            )

        return True, ""

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

    @classmethod
    def _build_init_kwargs(cls, paddle_lang: str, use_angle_cls: bool = True) -> dict:
        """Build PaddleOCR constructor kwargs compatible with the installed version.

        PaddleOCR 2.x: accepts show_log=False
        PaddleOCR 3.x: show_log removed; logging is suppressed via logging module
        We use inspect to detect which parameters are accepted so the code works
        with both versions without hard-coding a version number.
        """
        from paddleocr import PaddleOCR

        kwargs: dict = {"use_angle_cls": use_angle_cls, "lang": paddle_lang}

        try:
            sig = inspect.signature(PaddleOCR.__init__)
            if "show_log" in sig.parameters:
                kwargs["show_log"] = False
            else:
                logger.debug(
                    "PaddleOCR.__init__ does not accept show_log "
                    "(likely v3.x) – suppressing logging via logging module"
                )
        except Exception as exc:
            logger.debug(f"Could not inspect PaddleOCR.__init__ signature: {exc}")
            # Omit show_log safely; we suppress via logging module anyway

        return kwargs

    @classmethod
    def _init_ocr(cls, paddle_lang: str) -> None:
        """Initialize singleton PaddleOCR instance.

        - Suppresses ppocr/paddle logging via Python logging module
        - Uses inspect to pass show_log=False only when the parameter exists
        - Caches any initialization error in _init_error so that subsequent
          calls fail immediately instead of retrying (and spamming the log)
        """
        if cls._ocr_instance is not None:
            return
        if cls._init_error is not None:
            raise RuntimeError(cls._init_error)

        from paddleocr import PaddleOCR

        _suppress_paddle_logs()

        version = _get_paddleocr_version()
        logger.info(f"Initializing PaddleOCR (version={version}, lang={paddle_lang})")

        kwargs = cls._build_init_kwargs(paddle_lang)

        try:
            cls._ocr_instance = PaddleOCR(**kwargs)
            logger.info(f"PaddleOCR initialized successfully (version={version})")
        except TypeError as exc:
            # Safety net: if show_log still slipped through (e.g. via **kwargs
            # forwarding in some PaddleOCR version), retry without it.
            if "show_log" in str(exc):
                logger.warning(
                    f"PaddleOCR rejected show_log argument (version={version}): {exc}. "
                    "Retrying without show_log."
                )
                kwargs.pop("show_log", None)
                try:
                    cls._ocr_instance = PaddleOCR(**kwargs)
                    logger.info(
                        f"PaddleOCR initialized after removing show_log (version={version})"
                    )
                    return
                except Exception as exc2:
                    cls._init_error = str(exc2)
                    logger.error(
                        f"PaddleOCR initialization failed (version={version}): {exc2}"
                    )
                    raise RuntimeError(cls._init_error) from exc2
            else:
                cls._init_error = str(exc)
                logger.error(
                    f"PaddleOCR initialization failed (version={version}): {exc}"
                )
                raise RuntimeError(cls._init_error) from exc
        except Exception as exc:
            error_str = str(exc)
            # If failure is related to doc_orientation model (PP-LCNet) or
            # PaddlePaddle version mismatch (set_optimization_level), retry
            # with use_angle_cls=False to bypass the problematic model.
            # Novel OCR typically doesn't need angle classification.
            _retry_hints = ("PP-LCNet", "doc_ori", "set_optimization_level", "inference.yml")
            if any(hint in error_str for hint in _retry_hints):
                logger.warning(
                    f"PaddleOCR init failed with angle_cls (version={version}): {exc}. "
                    "Retrying with use_angle_cls=False (novel OCR doesn't require angle classification)..."
                )
                kwargs_no_angle = cls._build_init_kwargs(paddle_lang, use_angle_cls=False)
                try:
                    cls._ocr_instance = PaddleOCR(**kwargs_no_angle)
                    logger.info(
                        f"PaddleOCR initialized with use_angle_cls=False (version={version})"
                    )
                    return
                except Exception as exc2:
                    cls._init_error = str(exc2)
                    logger.error(
                        f"PaddleOCR fallback init (no angle_cls) also failed (version={version}): {exc2}"
                    )
                    raise RuntimeError(cls._init_error) from exc2
            else:
                cls._init_error = error_str
                logger.error(
                    f"PaddleOCR initialization failed (version={version}): {exc}"
                )
                raise RuntimeError(cls._init_error) from exc

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
            ocr = PaddleOCREngine._ocr_instance

            result = ocr.ocr(str(image_path), cls=True)
            if not result or not result[0]:
                warnings.append(f"PaddleOCR returned empty result: {image_path.name}")
                return "", warnings

            lines = []
            for line_info in result[0]:
                if line_info and len(line_info) >= 2:
                    text = (
                        line_info[1][0]
                        if isinstance(line_info[1], (list, tuple))
                        else str(line_info[1])
                    )
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
