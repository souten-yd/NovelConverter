"""Input ingestion layer – normalize txt/archive/epub/image uploads into UTF-8 text."""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.shared.logger import get_logger

logger = get_logger("ingest")

# BOM variants
_BOM_MAP = {
    b"\xef\xbb\xbf": "utf-8-sig",
    b"\xff\xfe": "utf-16-le",
    b"\xfe\xff": "utf-16-be",
}

TEXT_EXTENSIONS = {".txt", ".md"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
UPLOAD_EXTENSIONS = TEXT_EXTENSIONS | IMAGE_EXTENSIONS | {".zip", ".rar", ".epub"}
ARCHIVE_EXTENSIONS = {".zip", ".rar"}
SUPPORTED_NESTED_EXTENSIONS = TEXT_EXTENSIONS | IMAGE_EXTENSIONS | {".epub"}

MAX_ARCHIVE_FILES = 500
MAX_ARCHIVE_TOTAL_BYTES = 500 * 1024 * 1024
MAX_ARCHIVE_NESTED_DEPTH = 8


def _compact_warnings(
    warnings: list[str],
    *,
    detail_limit: int = 3,
) -> list[str]:
    """Limit repeated OCR warnings to avoid huge per-page traceback spam."""
    if not warnings:
        return []
    unique: list[str] = []
    counts: dict[str, int] = {}
    for w in warnings:
        key = (w or "").strip()
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
        if key not in unique:
            unique.append(key)
    compact = unique[:detail_limit]
    shown = sum(counts[w] for w in compact)
    hidden = sum(counts.values()) - shown
    if hidden > 0:
        compact.append(f"{hidden} additional warnings aggregated")
    return compact


@dataclass
class IngestedText:
    relative_path: str
    kind: str
    text: str
    status: str = "ok"
    warning: str | None = None


@dataclass
class IngestSummary:
    original_upload_name: str
    detected_input_type: str
    source_type: str
    extracted_files: list[dict[str, Any]]
    warnings: list[str]
    char_count: int
    normalized_filename: str
    upload_size_bytes: int


def detect_encoding(raw: bytes) -> str:
    for bom, enc in _BOM_MAP.items():
        if raw.startswith(bom):
            return enc
    return "utf-8"


def load_text_file(path: Path) -> str:
    """Load a text file, handling common encodings and BOM."""
    raw = path.read_bytes()
    encoding = detect_encoding(raw)
    try:
        text = raw.decode(encoding)
    except UnicodeDecodeError:
        # fallback
        text = raw.decode("cp932", errors="replace")
        logger.warning(f"Fell back to cp932 for {path}")
    # strip BOM char if present
    text = text.lstrip("\ufeff")
    logger.info(f"Loaded {path} ({len(text)} chars, encoding={encoding})")
    return text


def save_project_text(project_dir: Path, text: str, filename: str = "uploaded_normalized.txt") -> Path:
    """Persist normalized UTF-8 text to project directory."""
    project_dir.mkdir(parents=True, exist_ok=True)
    dest = project_dir / filename
    dest.write_text(text, encoding="utf-8")
    return dest


def _sanitize_member_path(name: str) -> Path:
    p = Path(name)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"unsafe archive member path: {name}")
    return p


def _safe_extract_zip(archive_path: Path, dest_dir: Path, warnings: list[str]) -> list[Path]:
    extracted: list[Path] = []
    total = 0
    with zipfile.ZipFile(archive_path) as zf:
        infos = zf.infolist()
        if len(infos) > MAX_ARCHIVE_FILES:
            warnings.append(f"archive file count exceeds limit ({len(infos)} > {MAX_ARCHIVE_FILES}); truncating")
        for info in infos[:MAX_ARCHIVE_FILES]:
            if info.is_dir():
                continue
            try:
                safe_rel = _sanitize_member_path(info.filename)
            except ValueError as exc:
                warnings.append(str(exc))
                continue
            if len(safe_rel.parts) > MAX_ARCHIVE_NESTED_DEPTH:
                warnings.append(f"member depth too deep: {info.filename}")
                continue
            total += info.file_size
            if total > MAX_ARCHIVE_TOTAL_BYTES:
                warnings.append("archive total size exceeds limit; stopping extraction")
                break
            out_path = dest_dir / safe_rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, out_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(out_path)
    return extracted


def _safe_extract_rar(archive_path: Path, dest_dir: Path, warnings: list[str]) -> list[Path]:
    try:
        import rarfile
    except Exception:
        warnings.append("RAR backend missing")
        return []

    extracted: list[Path] = []
    total = 0
    try:
        with rarfile.RarFile(archive_path) as rf:
            infos = rf.infolist()
            if len(infos) > MAX_ARCHIVE_FILES:
                warnings.append(f"archive file count exceeds limit ({len(infos)} > {MAX_ARCHIVE_FILES}); truncating")
            for info in infos[:MAX_ARCHIVE_FILES]:
                if info.isdir():
                    continue
                try:
                    safe_rel = _sanitize_member_path(info.filename)
                except ValueError as exc:
                    warnings.append(str(exc))
                    continue
                if len(safe_rel.parts) > MAX_ARCHIVE_NESTED_DEPTH:
                    warnings.append(f"member depth too deep: {info.filename}")
                    continue
                total += info.file_size
                if total > MAX_ARCHIVE_TOTAL_BYTES:
                    warnings.append("archive total size exceeds limit; stopping extraction")
                    break
                out_path = dest_dir / safe_rel
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with rf.open(info) as src, out_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted.append(out_path)
    except Exception as exc:
        warnings.append(f"Failed to extract RAR: {exc}")
    return extracted


def extract_archive(archive_path: Path, dest_dir: Path) -> tuple[list[Path], list[str]]:
    warnings: list[str] = []
    dest_dir.mkdir(parents=True, exist_ok=True)
    suffix = archive_path.suffix.lower()
    if suffix == ".zip":
        files = _safe_extract_zip(archive_path, dest_dir, warnings)
    elif suffix == ".rar":
        files = _safe_extract_rar(archive_path, dest_dir, warnings)
    else:
        return [], [f"Unsupported archive type: {suffix}"]
    return files, warnings


def collect_supported_files(root_dir: Path) -> tuple[list[Path], list[dict[str, Any]], list[str]]:
    found: list[Path] = []
    extracted_files: list[dict[str, Any]] = []
    warnings: list[str] = []

    for path in sorted(p for p in root_dir.rglob("*") if p.is_file()):
        rel = str(path.relative_to(root_dir)).replace("\\", "/")
        ext = path.suffix.lower()
        if ext in SUPPORTED_NESTED_EXTENSIONS:
            found.append(path)
            kind = "txt" if ext in TEXT_EXTENSIONS else "epub" if ext == ".epub" else "image"
            extracted_files.append({"relative_path": rel, "kind": kind, "chars": 0, "status": "pending", "warning": None})
        else:
            extracted_files.append({"relative_path": rel, "kind": "unsupported", "chars": 0, "status": "skipped", "warning": "unsupported file"})
            warnings.append(f"Unsupported file skipped: {rel}")

    return found, extracted_files, warnings


def extract_epub_text(epub_path: Path) -> tuple[str, list[str]]:
    warnings: list[str] = []
    try:
        from ebooklib import epub
    except Exception:
        warnings.append("EbookLib not installed")
        return "", warnings

    try:
        from bs4 import BeautifulSoup
    except Exception:
        warnings.append("BeautifulSoup (bs4) not installed")
        return "", warnings

    try:
        book = epub.read_epub(str(epub_path))
        docs = list(book.get_items_of_type(epub.ITEM_DOCUMENT))
    except Exception as exc:
        return "", [f"EPUB parse failed: {exc}"]

    chunks: list[str] = []
    for idx, item in enumerate(docs, start=1):
        href = getattr(item, "file_name", None) or getattr(item, "get_name", lambda: f"chapter-{idx}")()
        lower = str(href).lower()
        if any(key in lower for key in ["toc", "nav", "contents"]):
            continue
        try:
            soup = BeautifulSoup(item.get_content(), "html.parser")
            for bad in soup(["script", "style", "nav"]):
                bad.extract()
            text = soup.get_text("\n", strip=True)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
        except Exception as exc:
            warnings.append(f"EPUB item parse failed: {href}: {exc}")
            continue
        if not text:
            continue
        chunks.append(f"===== EPUB: {epub_path.name} / Chapter {idx} =====\n{text}")

    if not chunks:
        warnings.append(f"No EPUB text extracted: {epub_path.name}")
    return "\n\n".join(chunks), warnings



def extract_image_text(
    image_path: Path,
    ocr_lang: str = "jpn+eng",
    ocr_engine: str = "tesseract",
) -> tuple[str, list[str]]:
    """Extract text from an image using the specified OCR engine.

    Falls back to tesseract if the specified engine is unavailable.
    """
    from app.orchestrator.services.ocr.factory import get_engine

    engine = get_engine(ocr_engine)
    available, missing = engine.is_available()
    if not available:
        logger.warning(f"OCR engine '{ocr_engine}' unavailable ({missing}), falling back to tesseract")
        engine = get_engine("tesseract")
        available2, missing2 = engine.is_available()
        if not available2:
            return "", [f"OCRエンジン '{ocr_engine}' が利用不可: {missing}", f"フォールバック先の Tesseract も利用不可: {missing2}"]

    return engine.extract_text(image_path, lang=ocr_lang)


def extract_images_parallel(
    image_paths: list[Path],
    ocr_engine: str = "tesseract",
    progress_cb: Any = None,
) -> tuple[list[tuple[Path, str, list[str]]], list[str]]:
    """Extract text from multiple images using the enhanced OCR pipeline.

    Uses the new parallel pipeline when multiple images are provided and
    a pipeline-compatible engine is selected.  Falls back to sequential
    processing otherwise.

    Returns:
        ([(path, text, warnings), ...], global_warnings)
    """
    if len(image_paths) < 2 or ocr_engine == "tesseract":
        # Sequential fallback for single images or Tesseract
        results = []
        warnings_all: list[str] = []
        for i, p in enumerate(image_paths):
            text, w = extract_image_text(p, ocr_engine=ocr_engine)
            compact_w = _compact_warnings(w)
            results.append((p, text, compact_w))
            warnings_all.extend(w)
            if progress_cb:
                try:
                    progress_cb("ocr_page", page=i + 1, total_pages=len(image_paths))
                except Exception:
                    pass
        return results, _compact_warnings(warnings_all)

    # Use the enhanced pipeline
    try:
        from app.orchestrator.services.ocr.pipeline import run_pipeline_for_ingest

        def _pipeline_cb(stage, cur, total, pct):
            if progress_cb and stage == "ocr":
                try:
                    progress_cb("ocr_page", page=cur, total_pages=total)
                except Exception:
                    pass

        pipeline_result = run_pipeline_for_ingest(
            image_paths,
            ocr_engine=ocr_engine,
            progress_cb=_pipeline_cb,
        )

        results = []
        warnings_all: list[str] = []
        # Build path → result mapping, sorted by page_index
        for page in sorted(pipeline_result.pages, key=lambda p: p.page_index):
            idx = page.page_index
            path = image_paths[idx] if idx < len(image_paths) else Path(page.image_path)
            text = page.plain_text
            w = page.warnings
            if page.status == "error":
                w = [page.error_message] + w
            w = _compact_warnings(w)
            # Add the OCR header for compatibility with existing flow
            if text:
                text = f"===== OCR: {path.name} =====\n{text}"
            results.append((path, text, w))
            warnings_all.extend(w)

        return results, _compact_warnings(warnings_all)

    except Exception as exc:
        logger.warning(f"Enhanced pipeline failed, falling back to sequential: {exc}")
        results = []
        warnings_all = [f"Enhanced pipeline error: {exc}"]
        for i, p in enumerate(image_paths):
            text, w = extract_image_text(p, ocr_engine=ocr_engine)
            compact_w = _compact_warnings(w)
            results.append((p, text, compact_w))
            warnings_all.extend(w)
        return results, _compact_warnings(warnings_all)


def build_combined_text(parts: list[IngestedText]) -> str:
    ordered = sorted(parts, key=lambda p: p.relative_path)
    # Prefer fully-ok items; fall back to warning-status items so partial OCR
    # results are not silently discarded.
    ok_body = [p.text.strip() for p in ordered if p.text and p.status == "ok"]
    if ok_body:
        return "\n\n".join(x for x in ok_body if x).strip() + "\n"
    warn_body = [p.text.strip() for p in ordered if p.text and p.status not in ("skipped", "error")]
    return "\n\n".join(x for x in warn_body if x).strip() + "\n"


def _strip_ocr_header(text: str) -> str:
    src = text or ""
    if src.startswith("===== OCR:") or src.startswith("===== EPUB:"):
        newline_pos = src.find("\n")
        if newline_pos >= 0:
            return src[newline_pos + 1:]
    return src


def _payload_chars(text: str) -> int:
    payload = _strip_ocr_header(text).strip()
    return len(payload)


def write_ingest_manifest(project_dir: Path, manifest: dict[str, Any]) -> Path:
    path = project_dir / "ingest_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _load_text_bytes(content: bytes, filename: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix or ".txt") as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    try:
        return load_text_file(tmp_path)
    finally:
        os.unlink(tmp_path)


def ingest_uploaded_file(
    upload_file: Any,
    project_dir: Path,
    temp_root: Path,
    progress_cb: Any = None,  # optional callable(stage, **kwargs) → None
    ocr_engine: str = "tesseract",  # OCR engine to use
) -> IngestSummary:
    def _cb(stage: str, **kwargs: Any) -> None:
        if progress_cb is not None:
            try:
                progress_cb(stage, **kwargs)
            except Exception:
                pass  # never let progress tracking break ingest

    filename = upload_file.filename or "upload.bin"
    ext = Path(filename).suffix.lower()
    if ext not in UPLOAD_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {ext}")

    raw = upload_file.file.read()
    extracted_files: list[dict[str, Any]] = []
    warnings: list[str] = []
    ingested: list[IngestedText] = []

    temp_root.mkdir(parents=True, exist_ok=True)
    _cb("upload")

    if ext in TEXT_EXTENSIONS:
        text = _load_text_bytes(raw, filename)
        ingested.append(IngestedText(relative_path=filename, kind="txt", text=text))
        extracted_files.append({"relative_path": filename, "kind": "txt", "chars": len(text), "status": "ok", "warning": None})
        detected = source_type = "txt"
    elif ext == ".epub":
        _cb("unpack")
        epub_path = temp_root / filename
        epub_path.write_bytes(raw)
        _cb("page_scan")
        text, w = extract_epub_text(epub_path)
        for wi in w:
            _cb("ocr_page", warning=wi)
        warnings.extend(w)
        status = "ok" if text else "warning"
        ingested.append(IngestedText(relative_path=filename, kind="epub", text=text, status="ok" if text else "skipped", warning="; ".join(w) if w else None))
        extracted_files.append({"relative_path": filename, "kind": "epub", "chars": _payload_chars(text), "status": status, "warning": "; ".join(w) if w else None})
        detected = source_type = "epub"
    elif ext in IMAGE_EXTENSIONS:
        _cb("page_scan", total_pages=1)
        image_path = temp_root / filename
        image_path.write_bytes(raw)
        _cb("ocr_page", page=1, total_pages=1)
        text, w = extract_image_text(image_path, ocr_engine=ocr_engine)
        warnings.extend(w)
        status = "ok" if text else "warning"
        ingested.append(IngestedText(relative_path=filename, kind="image", text=text, status="ok" if text else "skipped", warning="; ".join(w) if w else None))
        extracted_files.append({"relative_path": filename, "kind": "image", "chars": _payload_chars(text), "status": status, "warning": "; ".join(w) if w else None})
        detected = source_type = "image"
    else:
        _cb("unpack")
        detected = source_type = "archive"
        archive_path = temp_root / filename
        archive_path.write_bytes(raw)
        extracted, archive_warnings = extract_archive(archive_path, temp_root / "extracted")
        warnings.extend(archive_warnings)

        supported_paths, manifest_entries, collect_warnings = collect_supported_files(temp_root / "extracted")
        warnings.extend(collect_warnings)
        extracted_files.extend(manifest_entries)

        # Determine how many image/epub files we have for progress tracking
        image_files = [p for p in supported_paths if p.suffix.lower() in IMAGE_EXTENSIONS]
        total_pages = len(image_files) if image_files else len(supported_paths)
        _cb("page_scan", total_pages=total_pages)

        entry_by_path = {entry["relative_path"]: entry for entry in extracted_files}
        extracted_root = temp_root / "extracted"
        processed_count = 0

        # Separate image and non-image files
        image_supported = [p for p in sorted(supported_paths) if p.suffix.lower() in IMAGE_EXTENSIONS]
        non_image_supported = [p for p in sorted(supported_paths) if p.suffix.lower() not in IMAGE_EXTENSIONS]

        # Process non-image files sequentially
        for path in non_image_supported:
            rel_key = str(path.relative_to(extracted_root)).replace("\\", "/")
            rel_entry = entry_by_path.get(rel_key)
            if not rel_entry:
                continue

            text = ""
            proc_warnings: list[str] = []
            pext = path.suffix.lower()
            if pext in TEXT_EXTENSIONS:
                text = load_text_file(path)
                rel_entry["kind"] = "txt"
            elif pext == ".epub":
                _cb("ocr_page", page=processed_count + 1, total_pages=total_pages)
                text, proc_warnings = extract_epub_text(path)
                rel_entry["kind"] = "epub"

            warnings.extend(proc_warnings)
            rel_entry["chars"] = _payload_chars(text)
            rel_entry["status"] = "ok" if text else "warning"
            rel_entry["warning"] = "; ".join(proc_warnings) if proc_warnings else None

            ingested.append(
                IngestedText(
                    relative_path=rel_entry["relative_path"],
                    kind=rel_entry["kind"],
                    text=text,
                    status="ok" if text else "skipped",
                    warning=rel_entry["warning"],
                )
            )

        # Process image files using parallel pipeline
        if image_supported:
            ocr_results, ocr_warnings = extract_images_parallel(
                image_supported,
                ocr_engine=ocr_engine,
                progress_cb=progress_cb,
            )
            warnings.extend(ocr_warnings)

            for path, text, proc_warnings in ocr_results:
                rel_key = str(path.relative_to(extracted_root)).replace("\\", "/")
                rel_entry = entry_by_path.get(rel_key)
                if not rel_entry:
                    continue

                rel_entry["kind"] = "image"
                for wi in proc_warnings:
                    _cb("ocr_page", page=processed_count, total_pages=total_pages, warning=wi)
                rel_entry["chars"] = _payload_chars(text)
                rel_entry["status"] = "ok" if text else "warning"
                rel_entry["warning"] = "; ".join(proc_warnings) if proc_warnings else None

                ingested.append(
                    IngestedText(
                        relative_path=rel_entry["relative_path"],
                        kind=rel_entry["kind"],
                        text=text,
                        status="ok" if text else "skipped",
                        warning=rel_entry["warning"],
                    )
                )

    _cb("text_merge")
    combined = build_combined_text(ingested)
    _cb("normalize")
    normalized_filename = "uploaded_normalized.txt"
    save_project_text(project_dir, combined, filename=normalized_filename)

    processed_entries = [f for f in extracted_files if f.get("status") == "ok"]
    manifest = {
        "original_upload_name": filename,
        "detected_input_type": detected,
        "extracted_files": extracted_files,
        "warnings": warnings,
        "totals": {
            "files": len(extracted_files),
            "processed": len(processed_entries),
            "chars": sum(int(f.get("chars", 0) or 0) for f in processed_entries),
        },
    }
    write_ingest_manifest(project_dir, manifest)

    return IngestSummary(
        original_upload_name=filename,
        detected_input_type=detected,
        source_type=source_type,
        extracted_files=extracted_files,
        warnings=warnings,
        char_count=len(combined),
        normalized_filename=normalized_filename,
        upload_size_bytes=len(raw),
    )
