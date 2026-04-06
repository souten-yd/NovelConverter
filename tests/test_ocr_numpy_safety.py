from __future__ import annotations

import numpy as np

from app.orchestrator.services.ocr.executor import _dict_to_page_result, _populate_tokens_from_paddle
from app.orchestrator.services.ocr.models import OCRPageResult
from app.orchestrator.services.ocr.numpy_safety import (
    deep_to_py_scalars,
    normalize_image_for_ocr,
    safe_int,
)
from app.orchestrator.services.ocr.paddleocr_engine import PaddleOCREngine


def test_deep_to_py_scalars_converts_nested_numpy_scalars() -> None:
    src = {
        "a": np.int16(7),
        "b": [np.int32(2), {"c": np.float32(0.5)}],
        "d": (np.uint16(9), np.int64(3)),
    }
    out = deep_to_py_scalars(src)
    assert isinstance(out["a"], int)
    assert isinstance(out["b"][0], int)
    assert isinstance(out["b"][1]["c"], float)
    assert isinstance(out["d"][0], int)
    assert isinstance(out["d"][1], int)


def test_normalize_image_for_ocr_uint16_and_int16_and_float32() -> None:
    u16 = np.array([[0, 1024], [4096, 65535]], dtype=np.uint16)
    i16 = np.array([[-32768, 0], [1024, 32767]], dtype=np.int16)
    f32 = np.array([[0.0, 12.3], [255.9, np.nan]], dtype=np.float32)

    for arr in (u16, i16, f32):
        out = normalize_image_for_ocr(arr)
        assert out.dtype == np.uint8
        assert out.shape == arr.shape


def test_normalize_image_for_ocr_rgba_to_rgb_uint8() -> None:
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    out = normalize_image_for_ocr(rgba)
    assert out.dtype == np.uint8
    assert out.shape == (4, 4, 3)


def test_paddle_predict_result_and_executor_conversion_remove_numpy_scalars() -> None:
    raw = [{
        "rec_texts": ["abc"],
        "rec_scores": [np.float32(0.98)],
        "rec_polys": [np.array([[np.int16(1), np.int16(2)], [3, 2], [3, 6], [1, 6]])],
    }]
    normalized = PaddleOCREngine._normalize_predict_result(raw)
    page = OCRPageResult(page_index=0, image_path="dummy.png")
    _populate_tokens_from_paddle(page, normalized)
    assert page.tokens
    token = page.tokens[0]
    assert isinstance(token.confidence, float)
    assert isinstance(token.bbox.x_min, float)

    as_dict = {
        "page_index": np.int16(1),
        "engine": "paddle_fast",
        "elapsed_ms": np.int32(15),
        "plain_text": "abc",
        "tokens": [{
            "text": "abc",
            "bbox": [np.int16(1), np.int16(2), np.int16(3), np.int16(4)],
            "confidence": np.float32(0.95),
            "block_order": np.int16(0),
            "line_order": np.int16(1),
            "token_order": np.int16(2),
            "is_ruby_candidate": np.bool_(False),
        }],
    }
    restored = _dict_to_page_result(as_dict)
    assert isinstance(restored.page_index, int)
    assert isinstance(restored.elapsed_ms, int)
    assert isinstance(restored.tokens[0].line_order, int)
    assert isinstance(restored.tokens[0].confidence, float)


def test_regression_numpy_int16_bit_length_safe_int() -> None:
    value = np.int16(1024)
    assert safe_int(value).bit_length() == 11


def test_populate_tokens_skips_malformed_polygon_but_keeps_valid_items() -> None:
    page = OCRPageResult(page_index=0, image_path="dummy.png")
    raw_items = [
        {"text": "bad", "score": 0.5, "poly": [[1], [2], [3], [4]]},
        {"text": "good", "score": 0.9, "poly": [[0, 0], [10, 0], [10, 10], [0, 10]]},
    ]
    _populate_tokens_from_paddle(page, raw_items)
    assert len(page.tokens) == 1
    assert page.tokens[0].text == "good"


def test_paddle_normalize_predict_result_handles_single_polygon_shape() -> None:
    raw = [{
        "rec_texts": ["abc"],
        "rec_scores": [0.98],
        "rec_polys": [[0, 0], [10, 0], [10, 10], [0, 10]],
    }]
    normalized = PaddleOCREngine._normalize_predict_result(raw)
    assert len(normalized) == 1
    assert normalized[0]["text"] == "abc"
