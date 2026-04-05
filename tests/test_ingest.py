import io
import json
import zipfile
from pathlib import Path

from app.orchestrator.services import ingest


class DummyUpload:
    def __init__(self, filename: str, payload: bytes):
        self.filename = filename
        self.file = io.BytesIO(payload)


def test_txt_ingest(tmp_path: Path):
    upload = DummyUpload("sample.txt", "こんにちは".encode("utf-8"))
    summary = ingest.ingest_uploaded_file(upload, tmp_path / "project", tmp_path / "temp")
    saved = (tmp_path / "project" / summary.normalized_filename).read_text(encoding="utf-8")
    assert "こんにちは" in saved
    assert summary.source_type == "txt"


def test_epub_ingest(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(ingest, "extract_epub_text", lambda _: ("===== EPUB: book.epub / Chapter 1 =====\n本文", []))
    upload = DummyUpload("book.epub", b"dummy")
    summary = ingest.ingest_uploaded_file(upload, tmp_path / "project", tmp_path / "temp")
    assert summary.source_type == "epub"
    assert summary.char_count > 0


def test_image_ocr_ingest(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(ingest, "extract_image_text", lambda _p, ocr_lang="jpn+eng": ("===== OCR: img.png =====\ntext", []))
    upload = DummyUpload("img.png", b"img")
    summary = ingest.ingest_uploaded_file(upload, tmp_path / "project", tmp_path / "temp")
    assert summary.source_type == "image"
    assert any(e["kind"] == "image" for e in summary.extracted_files)


def test_zip_ingest_mixed(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(ingest, "extract_epub_text", lambda _: ("epub text", []))
    monkeypatch.setattr(ingest, "extract_image_text", lambda _p, ocr_lang="jpn+eng": ("ocr text", []))

    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as zf:
        zf.writestr("a.txt", "alpha")
        zf.writestr("b.epub", b"epub")
        zf.writestr("c.png", b"png")
        zf.writestr("ignore.bin", b"x")

    upload = DummyUpload("bundle.zip", payload.getvalue())
    summary = ingest.ingest_uploaded_file(upload, tmp_path / "project", tmp_path / "temp")
    assert summary.source_type == "archive"
    assert summary.char_count > 0
    assert any("Unsupported file skipped" in w for w in summary.warnings)


def test_rar_backend_missing(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(ingest, "_safe_extract_rar", lambda *_: [])
    monkeypatch.setattr(ingest, "extract_archive", lambda _a, _d: ([], ["RAR backend missing"]))
    upload = DummyUpload("bundle.rar", b"rar")
    summary = ingest.ingest_uploaded_file(upload, tmp_path / "project", tmp_path / "temp")
    assert "RAR backend missing" in summary.warnings


def test_path_traversal_prevented(tmp_path: Path):
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as zf:
        zf.writestr("../evil.txt", "x")
        zf.writestr("ok.txt", "safe")

    upload = DummyUpload("unsafe.zip", payload.getvalue())
    summary = ingest.ingest_uploaded_file(upload, tmp_path / "project", tmp_path / "temp")
    assert any("unsafe archive member path" in w for w in summary.warnings)
    content = (tmp_path / "project" / summary.normalized_filename).read_text(encoding="utf-8")
    assert "safe" in content


def test_manifest_totals_count_only_ok_files(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        ingest,
        "extract_images_parallel",
        lambda _paths, **kwargs: (
            [
                (tmp_path / "temp" / "extracted" / "ok.png", "===== OCR: ok.png =====\n成功", []),
                (tmp_path / "temp" / "extracted" / "ng.png", "", ["NDLOCR-Lite failed (rc=2): ng.png"]),
            ],
            ["NDLOCR-Lite failed (rc=2): ng.png"],
        ),
    )

    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as zf:
        zf.writestr("ok.png", b"ok")
        zf.writestr("ng.png", b"ng")

    upload = DummyUpload("bundle.zip", payload.getvalue())
    ingest.ingest_uploaded_file(upload, tmp_path / "project", tmp_path / "temp", ocr_engine="ndlocr_lite")
    manifest = json.loads((tmp_path / "project" / "ingest_manifest.json").read_text(encoding="utf-8"))
    assert manifest["totals"]["processed"] == 1
    assert manifest["totals"]["chars"] == 2


def test_manifest_totals_zero_when_all_ocr_failed(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        ingest,
        "extract_images_parallel",
        lambda _paths, **kwargs: (
            [(tmp_path / "temp" / "extracted" / "only.png", "", ["tuple index out of range"])],
            ["tuple index out of range"],
        ),
    )

    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as zf:
        zf.writestr("only.png", b"img")

    upload = DummyUpload("bundle.zip", payload.getvalue())
    ingest.ingest_uploaded_file(upload, tmp_path / "project", tmp_path / "temp", ocr_engine="paddleocr")
    manifest = json.loads((tmp_path / "project" / "ingest_manifest.json").read_text(encoding="utf-8"))
    assert manifest["totals"]["processed"] == 0
    assert manifest["totals"]["chars"] == 0
