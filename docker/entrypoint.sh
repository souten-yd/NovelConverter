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
else
    echo "[entrypoint] WARNING: tesseract not found (OCR unavailable)"
fi

if command -v unrar >/dev/null 2>&1 || command -v 7z >/dev/null 2>&1 || command -v bsdtar >/dev/null 2>&1; then
    echo "[entrypoint] archive backend: unrar=$(command -v unrar || echo '-') 7z=$(command -v 7z || echo '-') bsdtar=$(command -v bsdtar || echo '-')"
else
    echo "[entrypoint] WARNING: no RAR backend command found (unrar/7z/bsdtar)"
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
