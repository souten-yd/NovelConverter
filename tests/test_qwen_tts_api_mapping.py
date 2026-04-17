from __future__ import annotations

import types

from app.shared.schemas import SynthesizeRequest
from app.workers.tts_base.synthesizer import Qwen3BaseSynthesizer
from app.workers.tts_custom.synthesizer import Qwen3CustomSynthesizer
from app.workers.tts_design.synthesizer import Qwen3DesignSynthesizer


class _ValidationResult:
    valid = True
    failure_reason = None
    rms = 0.1
    duration = 0.5


def _mock_soundfile(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "soundfile", types.SimpleNamespace(write=lambda *args, **kwargs: None))


def _mock_audio_validator(monkeypatch):
    monkeypatch.setitem(
        __import__("sys").modules,
        "app.shared.audio_validator",
        types.SimpleNamespace(validate_audio_array=lambda *args, **kwargs: _ValidationResult()),
    )


def test_custom_uses_generate_custom_voice(monkeypatch, tmp_path):
    _mock_soundfile(monkeypatch)
    _mock_audio_validator(monkeypatch)

    called = {}

    class _Model:
        def generate_custom_voice(self, **kwargs):
            called.update(kwargs)
            return [[0.0, 0.1, -0.1]], 22050

    synth = Qwen3CustomSynthesizer.__new__(Qwen3CustomSynthesizer)
    synth._model = _Model()
    synth._processor = object()
    synth._load_error = None
    synth._supported_speakers = ["ono_anna"]
    synth._supported_languages = ["japanese", "english"]

    req = SynthesizeRequest(text="こんにちは", speaker="ono_anna", language="ja", instruct="明るく", output_path=str(tmp_path / "c.wav"))
    res = synth.synthesize(req)

    assert res.success is True
    assert called == {"text": "こんにちは", "language": "japanese", "speaker": "ono_anna", "instruct": "明るく"}


def test_design_maps_voice_description_to_instruct(monkeypatch, tmp_path):
    _mock_soundfile(monkeypatch)
    _mock_audio_validator(monkeypatch)

    called = {}

    class _Model:
        def generate_voice_design(self, **kwargs):
            called.update(kwargs)
            return [[0.0, 0.1, -0.1]], 22050

    synth = Qwen3DesignSynthesizer.__new__(Qwen3DesignSynthesizer)
    synth._model = _Model()
    synth._processor = object()
    synth._load_error = None

    req = SynthesizeRequest(
        text="テスト",
        language="ja",
        voice_description="落ち着いた声",
        output_path=str(tmp_path / "d.wav"),
    )
    res = synth.synthesize(req)

    assert res.success is True
    assert called == {"text": "テスト", "language": "japanese", "instruct": "落ち着いた声"}


def test_clone_uses_generate_voice_clone(monkeypatch, tmp_path):
    _mock_soundfile(monkeypatch)
    _mock_audio_validator(monkeypatch)

    called = {}
    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(b"RIFF")

    class _Model:
        def generate_voice_clone(self, **kwargs):
            called.update(kwargs)
            return [[0.0, 0.1, -0.1]], 22050

    synth = Qwen3BaseSynthesizer.__new__(Qwen3BaseSynthesizer)
    synth._model = _Model()
    synth._processor = object()
    synth._load_error = None
    synth.MODEL_ID = "dummy"

    req = SynthesizeRequest(
        text="テスト",
        language="ja",
        reference_audio_path=str(ref_audio),
        reference_text="これは参照音声です",
        output_path=str(tmp_path / "b.wav"),
    )
    res = synth.synthesize(req)

    assert res.success is True
    assert called == {
        "text": "テスト",
        "language": "japanese",
        "ref_audio": str(ref_audio),
        "ref_text": "これは参照音声です",
    }


def test_clone_prompt_fallback_uses_generate_voice_clone(monkeypatch, tmp_path):
    _mock_soundfile(monkeypatch)
    _mock_audio_validator(monkeypatch)

    called = {}
    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(b"RIFF")

    class _Model:
        def create_voice_clone_prompt(self, **kwargs):
            called["prompt"] = kwargs
            return "PROMPT"

        def generate_voice_clone(self, **kwargs):
            called["generate"] = kwargs
            return [[0.0, 0.1, -0.1]], 24000

    synth = Qwen3BaseSynthesizer.__new__(Qwen3BaseSynthesizer)
    synth._model = _Model()
    synth._processor = object()
    synth._load_error = None
    synth.MODEL_ID = "dummy"

    req = SynthesizeRequest(
        text="テスト",
        language="ja",
        reference_audio_path=str(ref_audio),
        reference_text="これは参照音声です",
        output_path=str(tmp_path / "b2.wav"),
        extra={"use_voice_clone_prompt": True},
    )
    res = synth.synthesize(req)

    assert res.success is True
    assert called["prompt"] == {
        "ref_audio": str(ref_audio),
        "ref_text": "これは参照音声です",
    }
    assert called["generate"] == {
        "text": "テスト",
        "language": "japanese",
        "voice_clone_prompt": "PROMPT",
    }
