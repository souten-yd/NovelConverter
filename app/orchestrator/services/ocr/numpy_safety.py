"""NumPy boundary safety helpers for OCR pipeline.

Centralizes conversion rules so every OCR engine adapter and downstream
pipeline stage can work with:
  - Python-native scalars (no numpy scalar leakage)
  - JSON-safe nested payloads
  - uint8 image arrays with consistent channel layout
"""
from __future__ import annotations

from typing import Any

import numpy as np


def is_numpy_scalar(value: Any) -> bool:
    return isinstance(value, np.generic)


def to_py_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def safe_int(value: Any) -> int | None:
    value = to_py_scalar(value)
    if value is None:
        return None
    return int(value)


def safe_float(value: Any) -> float | None:
    value = to_py_scalar(value)
    if value is None:
        return None
    return float(value)


def safe_bool(value: Any) -> bool | None:
    value = to_py_scalar(value)
    if value is None:
        return None
    return bool(value)


def deep_to_py_scalars(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {deep_to_py_scalars(k): deep_to_py_scalars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [deep_to_py_scalars(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(deep_to_py_scalars(v) for v in obj)
    if isinstance(obj, set):
        return {deep_to_py_scalars(v) for v in obj}
    return obj


def safe_for_json(obj: Any) -> Any:
    return deep_to_py_scalars(obj)


def normalize_image_for_ocr(image: Any) -> np.ndarray:
    """Normalize arbitrary image-like input into uint8 OCR-ready array.

    Rules:
      - dtype normalized to uint8
      - grayscale: HxW kept as-is
      - RGB: HxWx3 kept as-is
      - RGBA: alpha dropped => HxWx3
    """
    arr = np.asarray(image)

    if arr.ndim not in (2, 3):
        raise TypeError(f"Unsupported image ndim for OCR: {arr.ndim}")

    if arr.ndim == 3:
        channels = arr.shape[2]
        if channels == 4:
            arr = arr[:, :, :3]
        elif channels == 3:
            pass
        elif channels == 1:
            arr = arr[:, :, 0]
        else:
            raise TypeError(f"Unsupported image channel count for OCR: {channels}")

    if arr.dtype == np.uint8:
        return arr

    if np.issubdtype(arr.dtype, np.integer):
        arr_f = arr.astype(np.float32)
        mn = float(np.min(arr_f))
        mx = float(np.max(arr_f))
        if mx <= mn:
            return np.zeros(arr.shape, dtype=np.uint8)
        return np.clip((arr_f - mn) * 255.0 / (mx - mn), 0, 255).astype(np.uint8)

    if np.issubdtype(arr.dtype, np.floating):
        arr_f = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
        return np.clip(arr_f, 0, 255).astype(np.uint8)

    raise TypeError(f"Unsupported image dtype for OCR: {arr.dtype}")
