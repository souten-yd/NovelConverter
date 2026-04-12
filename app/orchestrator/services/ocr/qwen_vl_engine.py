"""Qwen-VL vision-language model OCR engine adapter.

Supports two modes:
  1. Remote API  – set LLM_API_URL (and optionally LLM_API_KEY / LLM_MODEL)
  2. Local model – requires transformers + GPU; model is lazy-loaded on first use
                   (downloaded to HF_HOME on first inference)
"""
from __future__ import annotations

import os
import gc
import re
from pathlib import Path
from typing import Optional

from app.orchestrator.services.ocr.base import OCREngine
from app.shared.logger import get_logger


# ── OCR prompt ────────────────────────────────────────────────────────────────
# Strict instruction that discourages common VLM failure modes:
#   - Adding an intro sentence ("以下が抽出したテキストです: ...")
#   - Wrapping the output in code fences
#   - Translating / paraphrasing / summarising
#   - Describing the image layout instead of returning text
# The prompt is intentionally in Japanese because the primary use case is
# Japanese novel OCR, but it explicitly preserves non-Japanese characters too.
_OCR_PROMPT = (
    "あなたは正確なOCRエンジンです。"
    "この画像に含まれる全ての文字・記号を、画像上の読み順（日本語は縦書きなら右→左、"
    "横書きなら左→右）のまま、一字一句そのまま書き起こしてください。\n"
    "厳守事項:\n"
    "1. 前置き・説明・コードフェンスを付けない。返すのは書き起こしたテキストのみ。\n"
    "2. 翻訳・要約・言い換えをしない。ルビや句読点、半角英数字もそのまま保持。\n"
    "3. 行の区切りは改行で表現する。読めない箇所は空白を残さず省略する。\n"
    "4. 画像に文字が存在しない場合は、何も出力しない（空文字列を返す）。"
)


def _strip_vlm_wrappers(raw: str) -> str:
    """Clean common VLM response wrappers (code fences, intro phrases)."""
    text = (raw or "").strip()
    if not text:
        return ""

    # Strip ```lang ... ``` / ``` ... ``` fences
    fence = re.match(r"^```(?:[\w-]+)?\s*\n?(.*?)\n?```\s*$", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    # Drop a leading "intro" line if it matches well-known patterns
    intro_patterns = (
        r"^(以下|こちらが|これが).*?(抽出|テキスト|結果).*?[:：]\s*",
        r"^(sure|here(?:'s| is)|here are).*?[:：]\s*",
        r"^画像(?:内|から)?の(?:テキスト|文字).*?[:：]\s*",
    )
    for pat in intro_patterns:
        m = re.match(pat, text, flags=re.IGNORECASE)
        if m:
            text = text[m.end():].lstrip()
            break

    return text.strip()

logger = get_logger("ocr.qwen_vl")

# Lazy-loaded model + processor (local mode)
_local_model = None
_local_processor = None
_local_model_id: str = ""
_last_error: str = ""


def get_qwen_vl_runtime_status() -> dict:
    """Return a normalized runtime status snapshot for Qwen-VL."""
    mode = "remote_api" if os.environ.get("LLM_API_URL", "") else "local"
    loaded = _local_model is not None
    return {
        "engine": "qwen_vl",
        "status": "running" if loaded else "stopped",
        "loaded": loaded,
        "current_model": _local_model_id if loaded else None,
        "device": "cuda" if loaded else None,
        "memory": {},
        "idle_timer": {
            "seconds": None,
            "timeout_seconds": None,
            "remaining_seconds": None,
            "deadline_at": None,
        },
        "active_jobs": 0,
        "lease_total": 0,
        "pending_unload": False,
        "mode": mode,
        "last_error": _last_error or None,
    }


def release_qwen_vl_resources() -> dict:
    """Release local Qwen-VL model/processor memory if loaded."""
    global _local_model, _local_processor, _local_model_id
    was_loaded = _local_model is not None or _local_processor is not None
    _local_model = None
    _local_processor = None
    _local_model_id = ""
    gc.collect()
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass
    return {
        "requested": True,
        "released": True,
        "previously_loaded": was_loaded,
    }


class QwenVLEngine(OCREngine):
    engine_id = "qwen_vl"
    display_name = "Qwen-VL"
    description = "視覚LLMベースOCR。高精度。LLM_API_URLでリモートAPI利用可。GPU環境ではローカル推論も可。"

    def is_available(self) -> tuple[bool, str]:
        # Mode 1: Remote API – always available if URL is configured
        if os.environ.get("LLM_API_URL", ""):
            return True, ""

        # Mode 2: Local transformers inference
        missing = []
        try:
            import transformers  # noqa: F401
        except ImportError:
            missing.append("transformers (pip install transformers)")

        try:
            import torch  # noqa: F401
        except ImportError:
            missing.append("torch (pip install torch)")

        if missing:
            return (
                False,
                ", ".join(missing)
                + " (またはLLM_API_URLを設定してリモートAPI経由で利用)",
            )

        # transformers + torch are present; GPU availability is checked at
        # inference time so we report available even without a current GPU
        # (RunPod attaches the GPU at container start, not at build time).
        return True, ""

    def get_preprocessing_strategies(self) -> list[str]:
        return ["original"]  # VLM handles preprocessing internally

    def get_dependencies_info(self) -> dict:
        return {
            "python_packages": [
                "transformers>=4.45.0",
                "torch>=2.3.0",
                "accelerate>=0.30.0",
                "qwen-vl-utils",
            ],
            "system_packages": [],
            "notes": (
                "LLM_API_URL環境変数でリモートAPI経由でも利用可能。"
                "ローカル推論にはGPU(VRAM 10GB+)が必要。"
                "モデルは初回推論時に自動ダウンロードされます。"
            ),
        }

    def extract_text(
        self,
        image_path: Path,
        lang: str = "jpn+eng",
        preprocessing: Optional[str] = None,
    ) -> tuple[str, list[str]]:
        warnings: list[str] = []

        llm_url = os.environ.get("LLM_API_URL", "")
        if llm_url:
            return self._extract_via_api(image_path, llm_url, warnings)

        return self._extract_local(image_path, warnings)

    # ── Remote API ─────────────────────────────────────────────────────────────

    def _extract_via_api(
        self,
        image_path: Path,
        llm_url: str,
        warnings: list[str],
    ) -> tuple[str, list[str]]:
        """Use a vision-capable LLM API (OpenAI-compatible) for OCR."""
        import base64
        import requests

        try:
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")

            suffix = image_path.suffix.lower().lstrip(".")
            mime = {
                "jpg": "jpeg", "jpeg": "jpeg", "png": "png",
                "webp": "webp", "bmp": "bmp",
                "tiff": "tiff", "tif": "tiff",
            }.get(suffix, "jpeg")

            headers: dict = {"Content-Type": "application/json"}
            api_key = os.environ.get("LLM_API_KEY", "")
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            payload = {
                "model": os.environ.get("LLM_MODEL", "gpt-4o-mini"),
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{mime};base64,{img_b64}"
                                },
                            },
                            {"type": "text", "text": _OCR_PROMPT},
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
            raw = resp.json()["choices"][0]["message"]["content"]
            text = _strip_vlm_wrappers(raw if isinstance(raw, str) else "")

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

    # ── Local transformers inference ────────────────────────────────────────────

    def _extract_local(
        self,
        image_path: Path,
        warnings: list[str],
    ) -> tuple[str, list[str]]:
        """Local Qwen2-VL inference via HuggingFace transformers."""
        global _local_model, _local_processor, _local_model_id, _last_error

        try:
            import torch
        except ImportError:
            warnings.append("Qwen-VL local: torch not installed")
            return "", warnings

        try:
            from transformers import (
                Qwen2VLForConditionalGeneration,
                AutoProcessor,
            )
        except ImportError:
            warnings.append("Qwen-VL local: transformers not installed or too old (need >=4.45)")
            return "", warnings

        if not torch.cuda.is_available():
            warnings.append(
                "Qwen-VL local: GPU not available. "
                "LLM_API_URL を設定してリモートAPI経由で利用してください。"
            )
            return "", warnings

        model_id = os.environ.get("QWEN_VL_MODEL", "Qwen/Qwen2-VL-7B-Instruct")

        try:
            # Lazy-load model once
            if _local_model is None or _local_model_id != model_id:
                logger.info(f"Qwen-VL: loading model {model_id} (first use, may download)...")
                _local_processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
                _local_model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                )
                _local_model.eval()
                _local_model_id = model_id
                logger.info(f"Qwen-VL: model loaded: {model_id}")
        except Exception as exc:
            _last_error = str(exc)
            warnings.append(f"Qwen-VL model load failed ({model_id}): {exc}")
            logger.error(f"Qwen-VL model load error: {exc}")
            return "", warnings

        try:
            from PIL import Image

            image = Image.open(image_path).convert("RGB")

            # Build messages in the Qwen2-VL chat format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": _OCR_PROMPT},
                    ],
                }
            ]

            # Apply chat template
            text_prompt = _local_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Try qwen_vl_utils if available (handles image resizing etc.)
            try:
                from qwen_vl_utils import process_vision_info  # type: ignore

                image_inputs, video_inputs = process_vision_info(messages)
                inputs = _local_processor(
                    text=[text_prompt],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to("cuda")
            except ImportError:
                # Fallback: pass image directly
                inputs = _local_processor(
                    text=[text_prompt],
                    images=[image],
                    padding=True,
                    return_tensors="pt",
                ).to("cuda")

            with torch.no_grad():
                generated_ids = _local_model.generate(**inputs, max_new_tokens=1024)

            # Decode only the newly generated tokens
            generated_ids_trimmed = generated_ids[:, inputs["input_ids"].shape[1]:]
            raw = _local_processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            output = _strip_vlm_wrappers(raw)

            if not output:
                warnings.append(f"Qwen-VL local returned empty result: {image_path.name}")
            else:
                logger.info(f"Qwen-VL local ok: {image_path.name} → {len(output)} chars")
            _last_error = ""

            wrapped = f"===== OCR: {image_path.name} =====\n{output}" if output else ""
            return wrapped, warnings
        except Exception as exc:
            _last_error = str(exc)
            warnings.append(f"Qwen-VL local inference failed: {image_path.name}: {exc}")
            logger.error(f"Qwen-VL local inference error for {image_path.name}: {exc}")
            return "", warnings
