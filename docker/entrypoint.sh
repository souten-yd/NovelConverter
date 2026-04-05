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
echo "[entrypoint] TTS workers: BASE_USE_REAL=${TTS_BASE_USE_REAL:-false} CUSTOM_USE_REAL=${TTS_CUSTOM_USE_REAL:-false} DESIGN_USE_REAL=${TTS_DESIGN_USE_REAL:-false}  (true=real model, false=mock)"
echo "[entrypoint] LLM API:   ${LLM_API_URL:-(not set, rule-based only)}"
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="${PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK:-True}"
echo "[entrypoint] Paddle model source check disabled: ${PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK}"
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

try:
    import paddlex
    paddlex_version = getattr(paddlex, "__version__", "unknown")
    print(f"  [ok]      paddlex (version={paddlex_version})")
except ImportError:
    print(f"  [info]    paddlex: not installed (optional)")

# Check CUDA availability for PaddlePaddle
try:
    import paddle
    if hasattr(paddle, "is_compiled_with_cuda"):
        print(f"  [info]    CUDA compiled: {paddle.is_compiled_with_cuda()}")
    if hasattr(paddle, "device") and hasattr(paddle.device, "get_device"):
        print(f"  [info]    device: {paddle.device.get_device()}")
except Exception:
    pass

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

# ── PaddleOCR engine smoke test ───────────────────────────────────────────────
# Runs in a subprocess so _init_error state does NOT leak into the server process.
echo "[entrypoint] PaddleOCR engine smoke test:"
cd "${APP_DIR:-/workspace/NovelConverter}" 2>/dev/null || true
python3 - <<'PADDLESMOKE'
import sys
try:
    from app.orchestrator.services.ocr.paddleocr_engine import PaddleOCREngine
    PaddleOCREngine.configure(device="gpu:0", use_layout=False)
    engine = PaddleOCREngine()
    available, reason = engine.is_available()
    if available:
        try:
            PaddleOCREngine._init_ocr("japan")
            st = PaddleOCREngine.get_runtime_status()
            print(f"  [ok]      PaddleOCR engine initialised ({reason})")
            print(f"  [info]    initialized={st['initialized']} smoke_test_passed={st['smoke_test_passed']} warning={st['smoke_test_warning'] or '-'}")
            print(f"  [info]    last_error={st['last_error'] or '-'}")
        except Exception as init_exc:
            print(f"  [FAILED]  PaddleOCR init error: {init_exc}", file=sys.stderr)
    else:
        print(f"  [MISSING] PaddleOCR not available: {reason}", file=sys.stderr)
except Exception as e:
    print(f"  [ERROR]   PaddleOCR smoke test exception: {e}", file=sys.stderr)
PADDLESMOKE

# ── PaddleOCR model cache integrity check ───────────────────────────────────
# PP-LCNet_x1_0_doc_ori is used by PaddleX's document orientation classifier.
# If the directory exists but inference.yml is missing, the cache is incomplete
# and will cause PaddleOCR initialization to fail. Delete it so PaddleOCR can
# re-download cleanly on first use.
PADDLE_MODELS_BASE="/root/.paddlex/official_models"
for model_name in PP-LCNet_x1_0_doc_ori PP-ShiTuV2_det PP-ShiTuV2_rec; do
    model_dir="${PADDLE_MODELS_BASE}/${model_name}"
    if [ -d "${model_dir}" ] && [ ! -f "${model_dir}/inference.yml" ]; then
        echo "[entrypoint] WARNING: Incomplete PaddleX model cache: ${model_dir} (missing inference.yml)"
        echo "[entrypoint]   Removing incomplete directory for automatic re-download"
        rm -rf "${model_dir}"
    fi
done

# ── NDLOCR-Lite: model download & diagnostics ─────────────────────────────────
echo "[entrypoint] NDLOCR-Lite diagnostics:"
NDLOCR_MODEL_DIR="${NDLOCR_MODEL_DIR:-/workspace/ndlocr_models}"
NDLOCR_DOWNLOADER="${APP_DIR:-/workspace/NovelConverter}/scripts/download_ndlocr_models.py"
echo "[entrypoint] NDLOCR model dir: ${NDLOCR_MODEL_DIR}"
echo "[entrypoint] NDLOCR downloader: ${NDLOCR_DOWNLOADER}"

# Auto-install NDLOCR-Lite when missing (RunPod runtime recovery)
if ! command -v ndlocr-lite >/dev/null 2>&1 && ! command -v ndlocr >/dev/null 2>&1; then
    echo "[entrypoint] NDLOCR-Lite: CLI not found. Attempting runtime install..."
    if python3 -m pip install --no-cache-dir git+https://github.com/ndl-lab/ndlocr-lite.git; then
        hash -r
        if command -v ndlocr-lite >/dev/null 2>&1 || command -v ndlocr >/dev/null 2>&1; then
            echo "[entrypoint] NDLOCR-Lite: runtime install succeeded"
        else
            echo "[entrypoint] WARNING: NDLOCR-Lite runtime install completed but CLI is still missing"
        fi
    else
        echo "[entrypoint] WARNING: NDLOCR-Lite runtime install failed"
    fi
fi

NDLOCR_BIN=""
if command -v ndlocr-lite >/dev/null 2>&1; then
    NDLOCR_BIN="$(command -v ndlocr-lite)"
elif command -v ndlocr >/dev/null 2>&1; then
    NDLOCR_BIN="$(command -v ndlocr)"
fi

if [ -n "${NDLOCR_BIN}" ]; then
    echo "[entrypoint] NDLOCR-Lite: CLI found at ${NDLOCR_BIN}"

    # Verify CLI is responsive
    if "${NDLOCR_BIN}" --help >/dev/null 2>&1; then
        echo "[entrypoint] NDLOCR-Lite: CLI sanity check passed"
    else
        echo "[entrypoint] WARNING: NDLOCR-Lite --help failed – CLI may be broken"
    fi

    # Ensure model directory exists
    mkdir -p "${NDLOCR_MODEL_DIR}"

    # Check whether models are already present (any recognised weight file)
    model_count=$(find "${NDLOCR_MODEL_DIR}" \( -name "*.pth" -o -name "*.pt" -o -name "*.onnx" -o -name "*.pdparams" -o -name "*.bin" -o -name "*.npz" \) 2>/dev/null | wc -l)
    if [ "${model_count}" -gt 0 ]; then
        echo "[entrypoint] NDLOCR-Lite: ${model_count} model file(s) already present in ${NDLOCR_MODEL_DIR} – skipping download"
    else
        echo "[entrypoint] NDLOCR-Lite: No model files found in ${NDLOCR_MODEL_DIR}."
        echo "[entrypoint]   Attempting automatic model download..."
        if [ ! -f "${NDLOCR_DOWNLOADER}" ]; then
            echo "[entrypoint] WARNING: NDLOCR downloader script not found: ${NDLOCR_DOWNLOADER}"
            model_count=0
        elif python3 "${NDLOCR_DOWNLOADER}"; then
            model_count=$(find "${NDLOCR_MODEL_DIR}" \( -name "*.pth" -o -name "*.pt" -o -name "*.onnx" -o -name "*.pdparams" -o -name "*.bin" -o -name "*.npz" \) 2>/dev/null | wc -l)
        else
            model_count=0
        fi

        if [ "${model_count}" -gt 0 ]; then
            echo "[entrypoint] NDLOCR-Lite: model download completed (${model_count} file(s))"
        else
            echo "[entrypoint] WARNING: NDLOCR-Lite model download failed or no model files found."
            echo "[entrypoint]   NDLOCR-Lite engine will report unavailable until models are present."
            echo "[entrypoint]   Manual fallback: python3 ${NDLOCR_DOWNLOADER}"
        fi
    fi
else
    echo "[entrypoint] WARNING: NDLOCR-Lite CLI not found – NDLOCR-Lite engine will be unavailable"
    echo "[entrypoint]   To install: pip install git+https://github.com/ndl-lab/ndlocr-lite.git"
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

# ── Qwen3-TTS model preparation ─────────────────────────────────────────────
echo "[entrypoint] Preparing Qwen3-TTS models..."
cd "${APP_DIR:-/workspace/NovelConverter}" 2>/dev/null || true
python3 -c "
import sys
try:
    from app.shared.model_manager import ensure_all_models
    status = ensure_all_models()
    for k, v in status.items():
        tag = 'ok' if v else 'FAILED'
        print(f'  [entrypoint] Qwen3-TTS {k}: {tag}')
    if not all(status.values()):
        print('  [entrypoint] WARNING: Some Qwen3-TTS models failed to download', file=sys.stderr)
except Exception as e:
    print(f'  [entrypoint] Model preparation error: {e}', file=sys.stderr)
" 2>&1 || echo "[entrypoint] WARNING: Model preparation script failed"

# ── LLM (Gemma GGUF) preparation ────────────────────────────────────────────
echo "[entrypoint] Preparing LLM model..."
python3 -c "
import sys
try:
    from app.shared.llm_downloader import ensure_llm_model
    path = ensure_llm_model()
    print(f'  [entrypoint] LLM model: {path}')
except Exception as e:
    print(f'  [entrypoint] LLM download error: {e}', file=sys.stderr)
" 2>&1 || echo "[entrypoint] WARNING: LLM preparation script failed"

# ── If CMD args supplied, run those instead ───────────────────────────────────
if [[ $# -gt 0 ]]; then
    exec "$@"
fi

# ── Start supervisord ─────────────────────────────────────────────────────────
echo "[entrypoint] Starting supervisord..."
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/novelconverter.conf
