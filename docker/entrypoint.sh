#!/usr/bin/env bash
# =============================================================================
# NovelConverter – Docker entrypoint
# - Creates runtime directories
# - Waits for workers to be healthy before reporting ready
# - Starts supervisord (manages all 4 services)
# =============================================================================
set -euo pipefail

install_llama_server_fallback() {
    local install_bin="/opt/llama-cpp/bin/llama-server"
    local install_lib="/opt/llama-cpp/lib"
    local release_json asset_regex asset_url asset_name source_root

    echo "[entrypoint] llama-server not found. Trying runtime fallback download..."
    release_json="/tmp/llama_release_runtime.json"
    asset_regex='^llama\.cpp-b[0-9]+-cuda-12\.8\.tar\.gz$'

    if ! curl -fsSL "https://api.github.com/repos/ai-dock/llama.cpp-cuda/releases/latest" -o "${release_json}"; then
        echo "[entrypoint] WARNING: failed to fetch llama.cpp-cuda release metadata."
        return 1
    fi

    asset_url="$(jq -r --arg re "${asset_regex}" '.assets[] | select(.name | test($re)) | .browser_download_url' "${release_json}" | head -n1)"
    asset_name="$(jq -r --arg re "${asset_regex}" '.assets[] | select(.name | test($re)) | .name' "${release_json}" | head -n1)"
    if [[ -z "${asset_url}" || "${asset_url}" == "null" ]]; then
        echo "[entrypoint] WARNING: no matching CUDA 12.8 llama.cpp artifact found."
        rm -f "${release_json}"
        return 1
    fi

    mkdir -p /tmp/llama_extract_runtime "$(dirname "${install_bin}")" "${install_lib}"
    if ! curl -fL "${asset_url}" -o "/tmp/${asset_name}"; then
        echo "[entrypoint] WARNING: failed to download ${asset_name}."
        rm -rf /tmp/llama_extract_runtime "${release_json}" "/tmp/${asset_name}"
        return 1
    fi

    if ! tar -xzf "/tmp/${asset_name}" -C /tmp/llama_extract_runtime; then
        echo "[entrypoint] WARNING: failed to extract ${asset_name}."
        rm -rf /tmp/llama_extract_runtime "${release_json}" "/tmp/${asset_name}"
        return 1
    fi

    source_root="$(dirname "$(find /tmp/llama_extract_runtime -type f -name llama-server -perm -u+x | head -n1)")"
    if [[ -z "${source_root}" ]]; then
        echo "[entrypoint] WARNING: extracted archive does not contain executable llama-server."
        rm -rf /tmp/llama_extract_runtime "${release_json}" "/tmp/${asset_name}"
        return 1
    fi

    cp -a "${source_root}/llama-server" "${install_bin}"
    if [[ -f "${source_root}/llama-cli" ]]; then
        cp -a "${source_root}/llama-cli" /opt/llama-cpp/bin/llama-cli
    fi
    find "${source_root}" \( -type f -o -type l \) -name '*.so*' -exec cp -a {} "${install_lib}/" \;
    chmod +x "${install_bin}"
    export LLAMA_SERVER_BIN="${install_bin}"
    export LD_LIBRARY_PATH="${install_lib}:${LD_LIBRARY_PATH:-}"

    rm -rf /tmp/llama_extract_runtime "${release_json}" "/tmp/${asset_name}"
    echo "[entrypoint] Runtime fallback installed llama-server at ${install_bin}"
    return 0
}

ensure_llama_server_ready() {
    local configured="${LLAMA_SERVER_BIN:-llama-server}"
    local resolved=""

    if [[ -x "${configured}" ]]; then
        resolved="${configured}"
    elif command -v "${configured}" >/dev/null 2>&1; then
        resolved="$(command -v "${configured}")"
    fi

    if [[ -n "${resolved}" ]]; then
        export LLAMA_SERVER_BIN="${resolved}"
        echo "[entrypoint] llama-server: ${resolved}"
        return 0
    fi

    echo "[entrypoint] WARNING: llama-server was not found (configured='${configured}')."
    if install_llama_server_fallback; then
        return 0
    fi

    echo "[entrypoint] WARNING: llama fallback download failed. LLM segmentation will stay unavailable until llama-server is installed."
    return 1
}

echo "============================================="
echo " NovelConverter – Starting services"
echo "============================================="

# ── Runtime directory setup ───────────────────────────────────────────────────
mkdir -p \
    "${DATA_DIR:-/workspace/data}/projects" \
    "${DATA_DIR:-/workspace/data}/outputs" \
    "${DATA_DIR:-/workspace/data}/temp" \
    "${DATA_DIR:-/workspace/data}/references" \
    "${HF_HOME:-/workspace/hf_cache}" \
    /var/log/novelconverter

echo "[entrypoint] Data dir: ${DATA_DIR:-/workspace/data}"
echo "[entrypoint] HF cache: ${HF_HOME:-/workspace/hf_cache}"
echo "[entrypoint] Mock mode: BASE=${TTS_BASE_USE_REAL:-false} CUSTOM=${TTS_CUSTOM_USE_REAL:-false} DESIGN=${TTS_DESIGN_USE_REAL:-false}"
echo "[entrypoint] LLM API:   ${LLM_API_URL:-(not set, rule-based only)}"
ensure_llama_server_ready || true

echo "[entrypoint] Runtime diagnostics:"
if command -v nvidia-smi &>/dev/null; then
    echo "[entrypoint] nvidia-smi:"
    nvidia-smi || true
else
    echo "[entrypoint] nvidia-smi: command not found"
fi

if command -v tesseract >/dev/null 2>&1; then
    echo "[entrypoint] tesseract: $(command -v tesseract)"
    tesseract --version | head -n1 || true
    # List available language packs – warn if Japanese is missing
    tess_langs="$(tesseract --list-langs 2>&1 | tr '\n' ' ')"
    echo "[entrypoint] tesseract languages: ${tess_langs}"
    if ! echo "${tess_langs}" | grep -q "jpn"; then
        echo "[entrypoint] WARNING: tesseract Japanese language data (jpn) not found – OCR quality will be poor"
    fi
else
    echo "[entrypoint] WARNING: tesseract not found (OCR unavailable)"
fi

if command -v unrar >/dev/null 2>&1 || command -v 7z >/dev/null 2>&1 || command -v bsdtar >/dev/null 2>&1; then
    echo "[entrypoint] archive backend: unrar=$(command -v unrar || echo '-') 7z=$(command -v 7z || echo '-') bsdtar=$(command -v bsdtar || echo '-')"
else
    echo "[entrypoint] WARNING: no RAR backend command found (unrar/7z/bsdtar)"
fi

# ── Python package diagnostics ───────────────────────────────────────────────
echo "[entrypoint] Python package diagnostics:"
python3 - <<'PYCHECK'
import sys
checks = {
    "PIL (Pillow)":     "PIL",
    "pytesseract":      "pytesseract",
    "rarfile":          "rarfile",
    "ebooklib":         "ebooklib",
    "bs4 (BeautifulSoup)": "bs4",
}
all_ok = True
for label, mod in checks.items():
    try:
        __import__(mod)
        print(f"  [ok]      {label}")
    except ImportError:
        print(f"  [MISSING] {label}  ← pip install {mod.lower()}", file=sys.stderr)
        all_ok = False
if not all_ok:
    print("  WARNING: Some optional packages missing – archive/OCR/EPUB features may be unavailable",
          file=sys.stderr)
PYCHECK

# ── PaddleOCR diagnostics ─────────────────────────────────────────────────────
echo "[entrypoint] PaddleOCR diagnostics:"
python3 - <<'PADDLECHECK'
import sys
try:
    import paddleocr
    version = getattr(paddleocr, "__version__", "unknown")
    print(f"  [ok]      paddleocr (version={version})")
except ImportError as e:
    print(f"  [MISSING] paddleocr: {e}", file=sys.stderr)

try:
    import paddle
    paddle_version = getattr(paddle, "__version__", "unknown")
    print(f"  [ok]      paddlepaddle (version={paddle_version})")
except ImportError as e:
    print(f"  [MISSING] paddlepaddle: {e}", file=sys.stderr)

# Check that show_log is NOT required (version compatibility guard)
try:
    import inspect
    from paddleocr import PaddleOCR
    sig = inspect.signature(PaddleOCR.__init__)
    params = list(sig.parameters.keys())
    has_show_log = "show_log" in params
    print(f"  [info]    PaddleOCR.__init__ params: {params[:8]}{'...' if len(params)>8 else ''}")
    print(f"  [info]    show_log accepted: {has_show_log}")
except Exception as e:
    print(f"  [info]    Could not inspect PaddleOCR signature: {e}")
PADDLECHECK

# ── NDLOCR-Lite: model download & diagnostics ─────────────────────────────────
echo "[entrypoint] NDLOCR-Lite diagnostics:"
NDLOCR_MODEL_DIR="${NDLOCR_MODEL_DIR:-/workspace/ndlocr_models}"
echo "[entrypoint] NDLOCR model dir: ${NDLOCR_MODEL_DIR}"

if command -v ndlocr >/dev/null 2>&1; then
    echo "[entrypoint] NDLOCR-Lite: ndlocr command found at $(command -v ndlocr)"

    # Verify CLI is responsive
    if ndlocr --help >/dev/null 2>&1; then
        echo "[entrypoint] NDLOCR-Lite: CLI sanity check passed"
    else
        echo "[entrypoint] WARNING: ndlocr --help failed – CLI may be broken"
    fi

    # Ensure model directory exists
    mkdir -p "${NDLOCR_MODEL_DIR}"

    # Check whether models are already present (any recognised weight file)
    model_count=$(find "${NDLOCR_MODEL_DIR}" \( -name "*.pth" -o -name "*.pt" -o -name "*.onnx" -o -name "*.pdparams" -o -name "*.bin" -o -name "*.npz" \) 2>/dev/null | wc -l)
    if [ "${model_count}" -gt 0 ]; then
        echo "[entrypoint] NDLOCR-Lite: ${model_count} model file(s) already present in ${NDLOCR_MODEL_DIR} – skipping download"
    else
        echo "[entrypoint] NDLOCR-Lite: No model files found in ${NDLOCR_MODEL_DIR}. Attempting model setup..."
        python3 - <<NDLSETUP
import os, sys, subprocess, shutil

model_dir = os.environ.get("NDLOCR_MODEL_DIR", "/workspace/ndlocr_models")
os.makedirs(model_dir, exist_ok=True)

# Try ndlocr's own model-download mechanism if it exists.
# Different versions expose this differently; we probe several approaches.
downloaded = False

# Approach 1: ndlocr package may expose a download_models() function
try:
    import ndlocr
    for fn_name in ("download_models", "setup_model", "download"):
        fn = getattr(ndlocr, fn_name, None)
        if fn is not None:
            print(f"[entrypoint] NDLOCR-Lite: calling ndlocr.{fn_name}(model_dir={model_dir!r})")
            fn(model_dir)
            downloaded = True
            break
except Exception as exc:
    print(f"[entrypoint] NDLOCR-Lite: package-level model download failed: {exc}")

# Approach 2: ndlocr CLI --download-model flag
if not downloaded:
    binary = shutil.which("ndlocr")
    for flag in ("--download-model", "--setup", "--init"):
        try:
            proc = subprocess.run(
                [binary, flag, "--model_path", model_dir],
                capture_output=True, text=True, timeout=300,
            )
            if proc.returncode == 0:
                print(f"[entrypoint] NDLOCR-Lite: model download via {flag} succeeded")
                downloaded = True
                break
            else:
                stderr = (proc.stderr or "").strip()[:200]
                print(f"[entrypoint] NDLOCR-Lite: {flag} returned rc={proc.returncode}: {stderr}")
        except subprocess.TimeoutExpired:
            print(f"[entrypoint] NDLOCR-Lite: {flag} timed out")
        except Exception as exc:
            print(f"[entrypoint] NDLOCR-Lite: {flag} error: {exc}")

if not downloaded:
    print(
        "[entrypoint] WARNING: NDLOCR-Lite model auto-download could not be completed.\\n"
        f"  Please manually download models to: {model_dir}\\n"
        "  See: https://github.com/ndl-lab/ndlocr_cli for instructions."
    )
else:
    count = sum(1 for _ in __import__("pathlib").Path(model_dir).rglob("*")
                if _.is_file())
    print(f"[entrypoint] NDLOCR-Lite: model download complete ({count} files in {model_dir})")
NDLSETUP
    fi
else
    echo "[entrypoint] WARNING: ndlocr command not found – NDLOCR-Lite engine will be unavailable"
    echo "[entrypoint]   To install: pip install git+https://github.com/ndl-lab/ndlocr_cli.git"
fi

if [[ -x "${LLAMA_SERVER_BIN:-}" ]]; then
    echo "[entrypoint] ldd ${LLAMA_SERVER_BIN}:"
    ldd "${LLAMA_SERVER_BIN}" || true
    echo "[entrypoint] ${LLAMA_SERVER_BIN} --version:"
    "${LLAMA_SERVER_BIN}" --version || true
else
    echo "[entrypoint] llama-server diagnostics skipped: LLAMA_SERVER_BIN is not executable (${LLAMA_SERVER_BIN:-unset})"
fi

# ── GPU check ─────────────────────────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
    echo ""
    echo "[entrypoint] GPU info:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
        | sed 's/^/  /' || echo "  (nvidia-smi failed)"
else
    echo "[entrypoint] No GPU detected – running in CPU mode"
fi
echo ""

# ── If CMD args supplied, run those instead ───────────────────────────────────
if [[ $# -gt 0 ]]; then
    exec "$@"
fi

# ── Start supervisord ─────────────────────────────────────────────────────────
echo "[entrypoint] Starting supervisord..."
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/novelconverter.conf
