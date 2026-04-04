"""Qwen-VL vision-language model OCR engine adapter."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from app.orchestrator.services.ocr.base import OCREngine
from app.shared.logger import get_logger

logger = get_logger("ocr.qwen_vl")


class QwenVLEngine(OCREngine):
    engine_id = "qwen_vl"
    display_name = "Qwen-VL"
    description = "視覚LLMベースOCR。高精度だが処理が重い。GPU必須。LLM APIまたはローカル推論。"

    def is_available(self) -> tuple[bool, str]:
        # Qwen-VL can be used via:
        # 1. Remote API (LLM_API_URL with vision support)
        # 2. Local transformers model
        llm_url = os.environ.get("LLM_API_URL", "")
        if llm_url:
            return True, ""

        missing = []
        try:
            import transformers  # noqa: F401
        except ImportError:
            missing.append("transformers (pip install transformers)")
        try:
            import torch  # noqa: F401
            if not torch.cuda.is_available():
                missing.append("CUDA GPU (Qwen-VL requires GPU)")
        except ImportError:
            missing.append("torch (pip install torch)")

        if missing:
            return False, ", ".join(missing) + " (またはLLM_API_URLを設定してリモートAPI経由で利用)"
        return True, ""

    def get_preprocessing_strategies(self) -> list[str]:
        return ["original"]  # VLM handles preprocessing internally

    def get_dependencies_info(self) -> dict:
        return {
            "python_packages": ["transformers>=4.44.0", "torch>=2.3.0", "accelerate>=0.30.0"],
            "system_packages": [],
            "notes": "LLM_API_URL環境変数でリモートAPI経由でも利用可能。ローカル推論にはGPU(VRAM 16GB+)が必要。",
        }

    def extract_text(
        self,
        image_path: Path,
        lang: str = "jpn+eng",
        preprocessing: Optional[str] = None,
    ) -> tuple[str, list[str]]:
        warnings: list[str] = []

        # Try remote API first
        llm_url = os.environ.get("LLM_API_URL", "")
        if llm_url:
            return self._extract_via_api(image_path, llm_url, lang, warnings)

        # Fall back to local model
        return self._extract_local(image_path, lang, warnings)

    def _extract_via_api(
        self, image_path: Path, llm_url: str, lang: str, warnings: list[str]
    ) -> tuple[str, list[str]]:
        """Use a vision-capable LLM API for OCR."""
        import base64
        import json
        import requests

        try:
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")

            suffix = image_path.suffix.lower().lstrip(".")
            mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp", "bmp": "bmp", "tiff": "tiff", "tif": "tiff"}.get(suffix, "jpeg")

            headers = {"Content-Type": "application/json"}
            api_key = os.environ.get("LLM_API_KEY", "")
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            payload = {
                "model": os.environ.get("LLM_MODEL", "gpt-4o-mini"),
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{img_b64}"}},
                            {"type": "text", "text": "この画像に含まれるテキストをすべて抽出してください。テキストのみを返してください。"},
                        ],
                    }
                ],
                "max_tokens": 4096,
                "temperature": 0.1,
            }

            resp = requests.post(
                f"{llm_url.rstrip('/')}/chat/completions",
                json=payload,
                headers=headers,
                timeout=120,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()

            if not text:
                warnings.append(f"Qwen-VL API returned empty result: {image_path.name}")
            else:
                logger.info(f"Qwen-VL API ok: {image_path.name} → {len(text)} chars")

            wrapped = f"===== OCR: {image_path.name} =====\n{text}" if text else ""
            return wrapped, warnings
        except Exception as exc:
            warnings.append(f"Qwen-VL API failed: {image_path.name}: {exc}")
            logger.warning(f"Qwen-VL API error for {image_path.name}: {exc}")
            return "", warnings

    def _extract_local(
        self, image_path: Path, lang: str, warnings: list[str]
    ) -> tuple[str, list[str]]:
        """Local Qwen-VL inference (requires GPU + large model)."""
        warnings.append(
            "Qwen-VL ローカル推論は未実装です。LLM_API_URL を設定してリモートAPI経由で利用してください。"
        )
        return "", warnings
