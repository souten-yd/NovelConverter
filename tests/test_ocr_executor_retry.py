import pytest

pytest.importorskip("numpy")

from app.orchestrator.services.ocr.executor import _run_paddle_ocr


def test_run_paddle_ocr_retries_with_layout_when_fast_mode_is_empty(monkeypatch):
    from app.orchestrator.services.ocr import paddleocr_engine as paddle_module
    from app.orchestrator.services import resource_manager as rm_module

    class FakePaddleOCR:
        use_layout = False

        @classmethod
        def configure(cls, device: str = "gpu:0", use_layout: bool = False) -> None:
            cls.use_layout = use_layout

        def is_available(self):
            return True, ""

        def run_paddle_ocr(self, image_path: str, lang: str = "japan"):
            if self.use_layout:
                return [{"text": "縦書き本文", "score": 0.9}]
            return []

    monkeypatch.setattr(paddle_module, "PaddleOCREngine", FakePaddleOCR)
    monkeypatch.setattr(rm_module, "acquire_lease", lambda *args, **kwargs: None)
    monkeypatch.setattr(rm_module, "release_lease", lambda *args, **kwargs: None)

    result = _run_paddle_ocr("/tmp/page1.jpg", page_index=0, use_layout=False)

    assert result.plain_text == "縦書き本文"
    assert result.engine == "paddle_layout"
    assert any("recovered with layout retry" in w for w in result.warnings)
