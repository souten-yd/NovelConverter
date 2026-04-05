from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image

from app.orchestrator.services.ocr.ndlocr_lite_engine import NDLOCRLiteEngine
from app.orchestrator.services.ocr.paddleocr_engine import PaddleOCREngine


def test_paddle_normalize_predict_result_supports_legacy_tuple_shape() -> None:
    raw = [[
        [
            [[0, 0], [10, 0], [10, 10], [0, 10]],
            ("縦書きテスト", 0.91),
        ],
        [
            [[12, 0], [40, 0], [40, 10], [12, 10]],
            ("2行目", 0.88),
        ],
    ]]
    normalized = PaddleOCREngine._normalize_predict_result(raw)
    assert [x["text"] for x in normalized] == ["縦書きテスト", "2行目"]
    assert all(isinstance(x["score"], float) for x in normalized)


def test_ndlocr_extract_text_builds_required_cli_args(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: list[list[str]] = []

    def _fake_run(cmd, **kwargs):
        captured.append(cmd)
        out_dir = Path(cmd[cmd.index("--output") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "page.txt").write_text("ndlocr text", encoding="utf-8")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "app.orchestrator.services.ocr.ndlocr_lite_engine.get_ndlocr_status",
        lambda: {"runtime_ready": True, "device_request": "cpu", "error_message": ""},
    )
    monkeypatch.setattr(
        "app.orchestrator.services.ocr.ndlocr_lite_engine._python_binary",
        lambda: "python",
    )
    monkeypatch.setattr(
        "app.orchestrator.services.ocr.ndlocr_lite_engine._OCR_SCRIPT",
        tmp_path / "ocr.py",
    )
    monkeypatch.setattr(
        "app.orchestrator.services.ocr.ndlocr_lite_engine._MODEL_DIR",
        tmp_path / "model",
    )
    monkeypatch.setattr(
        "app.orchestrator.services.ocr.ndlocr_lite_engine._NDLOCR_ROOT",
        tmp_path,
    )
    monkeypatch.setattr(
        "app.orchestrator.services.ocr.ndlocr_lite_engine.subprocess.run",
        _fake_run,
    )

    img = tmp_path / "sample.png"
    Image.new("RGB", (8, 8), "white").save(img)
    text, warnings = NDLOCRLiteEngine().extract_text(img)

    assert text
    assert warnings == []
    cmd = captured[0]
    assert "--sourceimg" in cmd
    assert "--output" in cmd
    assert "--sourcedir" not in cmd


def test_ndlocr_extract_text_rc2_is_reported_as_argument_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def _fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=2,
            stdout="",
            stderr="usage: ocr.py ... --output OUTPUT",
        )

    monkeypatch.setattr(
        "app.orchestrator.services.ocr.ndlocr_lite_engine.get_ndlocr_status",
        lambda: {"runtime_ready": True, "device_request": "cpu", "error_message": ""},
    )
    monkeypatch.setattr(
        "app.orchestrator.services.ocr.ndlocr_lite_engine._python_binary",
        lambda: "python",
    )
    monkeypatch.setattr(
        "app.orchestrator.services.ocr.ndlocr_lite_engine._OCR_SCRIPT",
        tmp_path / "ocr.py",
    )
    monkeypatch.setattr(
        "app.orchestrator.services.ocr.ndlocr_lite_engine._MODEL_DIR",
        tmp_path / "model",
    )
    monkeypatch.setattr(
        "app.orchestrator.services.ocr.ndlocr_lite_engine._NDLOCR_ROOT",
        tmp_path,
    )
    monkeypatch.setattr(
        "app.orchestrator.services.ocr.ndlocr_lite_engine.subprocess.run",
        _fake_run,
    )

    img = tmp_path / "sample.png"
    Image.new("RGB", (8, 8), "white").save(img)
    text, warnings = NDLOCRLiteEngine().extract_text(img)

    assert text == ""
    assert any("argument error (rc=2)" in w for w in warnings)


def test_ndlocr_extract_text_missing_output_file_returns_warning(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def _fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "app.orchestrator.services.ocr.ndlocr_lite_engine.get_ndlocr_status",
        lambda: {"runtime_ready": True, "device_request": "cpu", "error_message": ""},
    )
    monkeypatch.setattr(
        "app.orchestrator.services.ocr.ndlocr_lite_engine._python_binary",
        lambda: "python",
    )
    monkeypatch.setattr(
        "app.orchestrator.services.ocr.ndlocr_lite_engine._OCR_SCRIPT",
        tmp_path / "ocr.py",
    )
    monkeypatch.setattr(
        "app.orchestrator.services.ocr.ndlocr_lite_engine._MODEL_DIR",
        tmp_path / "model",
    )
    monkeypatch.setattr(
        "app.orchestrator.services.ocr.ndlocr_lite_engine._NDLOCR_ROOT",
        tmp_path,
    )
    monkeypatch.setattr(
        "app.orchestrator.services.ocr.ndlocr_lite_engine.subprocess.run",
        _fake_run,
    )

    img = tmp_path / "sample.png"
    Image.new("RGB", (8, 8), "white").save(img)
    text, warnings = NDLOCRLiteEngine().extract_text(img)

    assert text == ""
    assert any("output missing" in w for w in warnings)
    assert any("returned empty result" in w for w in warnings)
