"""Layer 1 – Input normalization.

Converts PDF / ZIP / folder / single-image inputs into a uniform list of
``PageJob`` objects, each carrying a stable ``page_index`` that is the sole
ordering key for every downstream stage.

Sorting uses *natural order* so that ``1.jpg, 2.jpg, 10.jpg`` yields
page indices 0, 1, 2 (not 0, 2, 1 as a naive lexicographic sort would).
"""
from __future__ import annotations

import os
import re
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Optional

from app.shared.logger import get_logger
from app.orchestrator.services.ocr.models import PageJob

logger = get_logger("ocr.input_normalizer")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
PDF_EXTENSIONS = {".pdf"}


# ---------------------------------------------------------------------------
# Natural sort key
# ---------------------------------------------------------------------------

_NATURAL_RE = re.compile(r"(\d+)")


def _natural_sort_key(path: Path) -> list:
    """Sort key that treats embedded numbers numerically.

    ``["1.jpg", "2.jpg", "10.jpg"]`` → sorted as 1, 2, 10.
    Falls back to case-insensitive string comparison for non-numeric parts.
    """
    parts: list = []
    for segment in _NATURAL_RE.split(path.name.lower()):
        if segment.isdigit():
            parts.append((0, int(segment)))
        else:
            parts.append((1, segment))
    return parts


# ---------------------------------------------------------------------------
# PDF rasterization
# ---------------------------------------------------------------------------

def _rasterize_pdf(pdf_path: Path, output_dir: Path, dpi: int = 300) -> list[Path]:
    """Convert each PDF page to a JPEG image using pdf2image (poppler).

    Returns paths sorted in page order.
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        logger.warning("pdf2image not installed – cannot rasterize PDF")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    images = convert_from_path(str(pdf_path), dpi=dpi, fmt="jpeg")
    paths: list[Path] = []
    for idx, img in enumerate(images):
        out = output_dir / f"{idx:06d}.jpg"
        img.save(str(out), "JPEG")
        paths.append(out)
    logger.info(f"PDF rasterized: {pdf_path.name} → {len(paths)} pages @ {dpi}dpi")
    return paths


# ---------------------------------------------------------------------------
# ZIP extraction
# ---------------------------------------------------------------------------

def _extract_zip_images(zip_path: Path, output_dir: Path) -> list[Path]:
    """Extract image files from a ZIP, ignoring non-image entries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            ext = Path(info.filename).suffix.lower()
            if ext not in IMAGE_EXTENSIONS:
                continue
            # Sanitize path
            safe_name = Path(info.filename).name
            if not safe_name:
                continue
            out = output_dir / safe_name
            # Handle duplicates
            counter = 1
            while out.exists():
                stem = Path(safe_name).stem
                out = output_dir / f"{stem}_{counter}{ext}"
                counter += 1
            with zf.open(info) as src, out.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(out)
    return extracted


# ---------------------------------------------------------------------------
# Folder scan
# ---------------------------------------------------------------------------

def _scan_folder_images(folder: Path) -> list[Path]:
    """Recursively find all image files in a folder."""
    return [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_inputs(
    input_path: Path,
    *,
    job_id: Optional[str] = None,
    work_dir: Optional[Path] = None,
    dpi: int = 300,
) -> list[PageJob]:
    """Normalize any supported input into an ordered list of ``PageJob``.

    Supported inputs:
      - Single image file
      - Directory of images
      - ZIP archive containing images
      - PDF file

    The returned list is sorted in *natural page order* with stable
    ``page_index`` values starting from 0.

    Args:
        input_path: Path to the input (file or directory).
        job_id: Unique job identifier.  Auto-generated if not provided.
        work_dir: Working directory for temporary files.  Defaults to a
            system temp directory.
        dpi: Resolution for PDF rasterization.

    Returns:
        A list of ``PageJob`` sorted by ``page_index``.
    """
    if job_id is None:
        job_id = f"job-{uuid.uuid4().hex[:8]}"

    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix=f"ocr_{job_id}_"))
    else:
        work_dir.mkdir(parents=True, exist_ok=True)

    pages_dir = work_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    source_name = input_path.name

    # Determine input type and collect image paths
    image_paths: list[Path] = []

    if input_path.is_dir():
        image_paths = _scan_folder_images(input_path)
        logger.info(f"Folder scan: {input_path} → {len(image_paths)} images")

    elif input_path.suffix.lower() in PDF_EXTENSIONS:
        image_paths = _rasterize_pdf(input_path, pages_dir, dpi=dpi)

    elif input_path.suffix.lower() == ".zip":
        extracted_dir = work_dir / "zip_extracted"
        image_paths = _extract_zip_images(input_path, extracted_dir)
        logger.info(f"ZIP extraction: {input_path.name} → {len(image_paths)} images")

    elif input_path.suffix.lower() in IMAGE_EXTENSIONS:
        image_paths = [input_path]

    else:
        logger.warning(f"Unsupported input type: {input_path.suffix}")
        return []

    # Sort by natural order
    image_paths.sort(key=_natural_sort_key)

    # Build page jobs
    jobs: list[PageJob] = []
    for idx, img_path in enumerate(image_paths):
        jobs.append(PageJob(
            source_doc_id=job_id,
            source_name=source_name,
            page_index=idx,
            image_path=str(img_path),
        ))

    logger.info(f"Normalized input: {source_name} → {len(jobs)} page jobs")
    return jobs


def normalize_image_list(
    image_paths: list[Path],
    *,
    job_id: Optional[str] = None,
    source_name: str = "images",
) -> list[PageJob]:
    """Normalize a pre-existing list of image paths into ``PageJob`` objects.

    This is used when the caller already has extracted images (e.g. from
    the existing ingest pipeline) and only needs page_index assignment.
    """
    if job_id is None:
        job_id = f"job-{uuid.uuid4().hex[:8]}"

    sorted_paths = sorted(image_paths, key=_natural_sort_key)
    return [
        PageJob(
            source_doc_id=job_id,
            source_name=source_name,
            page_index=idx,
            image_path=str(p),
        )
        for idx, p in enumerate(sorted_paths)
    ]
