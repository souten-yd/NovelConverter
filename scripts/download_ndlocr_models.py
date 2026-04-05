#!/usr/bin/env python3
"""Download NDLOCR-Lite model files to the configured model directory.

NDLOCR-Lite (https://github.com/ndl-lab/ndlocr-lite) does not bundle model
weights in the pip package.  This script downloads the required ONNX models
from the official release assets and validates the result.

Usage:
    python3 scripts/download_ndlocr_models.py

Environment variables:
    NDLOCR_MODEL_DIR  Target directory (default: /workspace/ndlocr_models)
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_MODEL_DIR = "/workspace/ndlocr_models"

# NDLOCR-Lite official model archive URLs.
# These point to the ndl-lab GitHub releases.  Update the URLs when the
# upstream project publishes new model versions.
_MODEL_ARCHIVES: list[dict] = [
    {
        "name": "ndlocr-lite models",
        "url": "https://github.com/ndl-lab/ndlocr-lite/releases/download/v0.1.0/models.zip",
        "sha256": None,  # Set to expected hash once known, or None to skip verification
    },
]

# Extensions that indicate a valid model file
_MODEL_EXTENSIONS = {".pth", ".pt", ".onnx", ".pdparams", ".bin", ".npz"}
_AUX_EXTENSIONS = {".json", ".yaml", ".yml"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download(url: str, dest: Path, *, chunk_size: int = 1 << 20) -> None:
    """Download a URL to a local file with progress reporting."""
    print(f"  Downloading {url}")
    req = Request(url, headers={"User-Agent": "ndlocr-model-downloader/1.0"})
    with urlopen(req, timeout=120) as resp, open(dest, "wb") as f:
        total = resp.headers.get("Content-Length")
        downloaded = 0
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / int(total) * 100
                print(f"\r  {downloaded / 1e6:.1f} / {int(total) / 1e6:.1f} MB ({pct:.0f}%)", end="", flush=True)
            else:
                print(f"\r  {downloaded / 1e6:.1f} MB", end="", flush=True)
    print()


def _verify_sha256(path: Path, expected: str | None) -> bool:
    if expected is None:
        return True
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected:
        print(f"  ERROR: SHA-256 mismatch for {path.name}")
        print(f"    expected: {expected}")
        print(f"    actual:   {actual}")
        return False
    return True


def _extract(archive_path: Path, dest_dir: Path) -> None:
    """Extract a .zip or .tar.gz archive into dest_dir."""
    name = archive_path.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dest_dir)
    elif name.endswith(".tar.gz") or name.endswith(".tgz"):
        with tarfile.open(archive_path) as tf:
            tf.extractall(dest_dir)
    else:
        print(f"  WARNING: Unknown archive format: {archive_path.name}")


def _count_model_files(directory: Path) -> int:
    return sum(
        1 for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in _MODEL_EXTENSIONS
    )


def _count_aux_files(directory: Path) -> int:
    return sum(
        1 for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in _AUX_EXTENSIONS
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    model_dir = Path(os.environ.get("NDLOCR_MODEL_DIR", _DEFAULT_MODEL_DIR))
    print(f"NDLOCR-Lite model directory: {model_dir}")

    # Check if models already present
    existing = _count_model_files(model_dir)
    existing_aux = _count_aux_files(model_dir) if model_dir.exists() else 0
    if existing > 0 and existing_aux > 0:
        print(f"  already present, skipping download (models={existing}, aux={existing_aux})")
        return 0
    if model_dir.exists():
        print("  incomplete, re-downloading")
    else:
        print("  missing, downloading")

    # Clean up incomplete directory
    if model_dir.exists() and not any(model_dir.iterdir()):
        print("  Empty model directory found – will download fresh.")
    elif model_dir.exists():
        # Has files but no model files → might be incomplete
        print("  Directory exists but no model files found – removing for clean download.")
        shutil.rmtree(model_dir)

    model_dir.mkdir(parents=True, exist_ok=True)

    success = False
    for archive_info in _MODEL_ARCHIVES:
        print(f"\nDownloading: {archive_info['name']}")

        with tempfile.TemporaryDirectory(prefix="ndlocr_dl_") as tmpdir:
            tmp = Path(tmpdir)
            url = archive_info["url"]
            filename = url.rsplit("/", 1)[-1]
            archive_path = tmp / filename

            try:
                _download(url, archive_path)
            except Exception as exc:
                print(f"  ERROR: Download failed: {exc}")
                continue

            if not _verify_sha256(archive_path, archive_info.get("sha256")):
                continue

            print(f"  Extracting to {model_dir}")
            try:
                _extract(archive_path, model_dir)
                success = True
            except Exception as exc:
                print(f"  ERROR: Extraction failed: {exc}")
                continue

    # Validate result
    final_count = _count_model_files(model_dir)
    aux_count = _count_aux_files(model_dir)
    if final_count > 0 and aux_count > 0:
        print(f"\nSuccess: models={final_count}, aux={aux_count} in {model_dir}")
        return 0
    elif success:
        print(f"\nWARNING: Archive extracted but model set is incomplete in {model_dir}")
        print(f"  model_files={final_count}, aux_files={aux_count}")
        print("  Expected both weight files and metadata/config files.")
        return 1
    else:
        print(f"\nERROR: Model download failed. Please download manually:")
        print(f"  1. Visit https://github.com/ndl-lab/ndlocr-lite/releases")
        print(f"  2. Download the model archive")
        print(f"  3. Extract to {model_dir}")
        # Clean up empty directory
        if model_dir.exists() and not any(model_dir.iterdir()):
            model_dir.rmdir()
        return 1


if __name__ == "__main__":
    sys.exit(main())
