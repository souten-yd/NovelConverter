import json
from pathlib import Path

from app.orchestrator.routers.processing import _build_empty_segments_detail, _analyze_segmentation_stages
from app.orchestrator.services.preprocessor import preprocess


def test_build_empty_segments_detail_with_manifest_hints(tmp_path: Path):
    manifest = {
        "warnings": [
            "RAR backend missing",
            "OCR failed: page1.jpg: tesseract missing",
        ],
        "extracted_files": [
            {"relative_path": "page1.jpg", "status": "warning"},
            {"relative_path": "page2.jpg", "status": "warning"},
        ],
    }
    (tmp_path / "ingest_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    detail = _build_empty_segments_detail(tmp_path, raw_text="\n")

    assert "推定原因" in detail
    assert "正規化後テキストが空です" in detail
    assert "RAR展開バックエンド" in detail
    assert "OCR処理に失敗しました" in detail


def test_build_empty_segments_detail_without_hints(tmp_path: Path):
    detail = _build_empty_segments_detail(tmp_path, raw_text="本文あり")
    assert detail == "テキストからセグメントが生成できませんでした。ファイルの内容を確認してください。"


def test_analyze_segmentation_stages_empty_text():
    reasons, stats = _analyze_segmentation_stages("   \n\n")
    assert "empty text" in reasons
    assert stats["trimmed_chars"] == 0


def test_preprocess_fallback_non_empty_text():
    segments = preprocess("line1\nline2")
    assert len(segments) >= 1
