from __future__ import annotations


def test_tts_base_health_does_not_instantiate_synth(monkeypatch):
    from app.workers.tts_base import main as base_main

    class _Explode:
        def __init__(self):
            raise AssertionError("should not instantiate on /health")

    monkeypatch.setattr(base_main, "USE_REAL_MODEL", True)
    monkeypatch.setattr(base_main, "_synth", None)
    monkeypatch.setattr(base_main, "Qwen3BaseSynthesizer", _Explode)

    result = base_main.health()
    assert result.model_loaded is False


def test_tts_custom_health_does_not_instantiate_synth(monkeypatch):
    from app.workers.tts_custom import main as custom_main

    class _Explode:
        def __init__(self):
            raise AssertionError("should not instantiate on /health")

    monkeypatch.setattr(custom_main, "USE_REAL_MODEL", True)
    monkeypatch.setattr(custom_main, "_synth", None)
    monkeypatch.setattr(custom_main, "Qwen3CustomSynthesizer", _Explode)

    result = custom_main.health()
    assert result.model_loaded is False


def test_tts_design_health_does_not_instantiate_synth(monkeypatch):
    from app.workers.tts_design import main as design_main

    class _Explode:
        def __init__(self):
            raise AssertionError("should not instantiate on /health")

    monkeypatch.setattr(design_main, "USE_REAL_MODEL", True)
    monkeypatch.setattr(design_main, "_synth", None)
    monkeypatch.setattr(design_main, "Qwen3DesignSynthesizer", _Explode)

    result = design_main.health()
    assert result.model_loaded is False
