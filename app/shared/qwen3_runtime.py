"""Shared runtime helpers for Qwen3-TTS workers.

Centralises the load kwargs and generation defaults documented in the
upstream README (https://github.com/QwenLM/Qwen3-TTS) so the three
synthesizers (base / custom / design) stay consistent.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

# Upstream issues #10 / #211 describe the model failing to emit EOS on
# longer inputs, producing silent output or hanging. The README examples
# cap generation via `max_new_tokens`; 2048 covers typical narration
# segments while still bounding worst-case runtime.
DEFAULT_MAX_NEW_TOKENS = 2048


def _flash_attention_available() -> bool:
    try:
        import flash_attn  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def build_load_kwargs(torch_module: Any, device: str) -> Tuple[Dict[str, Any], str, str]:
    """Return ``(from_pretrained_kwargs, dtype_name, attn_backend)`` for ``device``.

    Mirrors the canonical README invocation::

        Qwen3TTSModel.from_pretrained(
            model_id,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

    Falls back to ``sdpa``/``eager`` when flash-attn is unavailable, and to
    fp32 / eager on CPU so workers still load without accelerators.
    """
    if device == "cuda":
        attn = "flash_attention_2" if _flash_attention_available() else "sdpa"
        kwargs: Dict[str, Any] = {
            "device_map": "cuda:0",
            "dtype": torch_module.bfloat16,
            "attn_implementation": attn,
        }
        return kwargs, "bfloat16", attn

    kwargs = {
        "dtype": torch_module.float32,
        "attn_implementation": "eager",
    }
    return kwargs, "float32", "eager"


def format_exception(exc: BaseException) -> str:
    """Return a non-empty error string for ``exc``.

    ``str(exc)`` is empty for bare ``Exception()`` / argless raises and is
    literally ``"()"`` when args is an empty tuple wrapped in a single
    positional argument — both have been observed leaking to the UI as
    ``生成失敗()``. Always include the exception class name.
    """
    message = str(exc).strip()
    if message and message != "()":
        return f"{type(exc).__name__}: {message}"
    return f"{type(exc).__name__} (no message)"
