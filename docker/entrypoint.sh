#!/usr/bin/env bash
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

    echo "[entrypoint] WARNING: llama fallback download failed."
    return 1
}

echo "============================================="
echo " NovelConverter – Starting services"
echo "============================================="

mkdir -p \
    "${DATA_DIR:-/workspace/data}/projects" \
    "${DATA_DIR:-/workspace/data}/outputs" \
    "${DATA_DIR:-/workspace/data}/temp" \
    "${DATA_DIR:-/workspace/data}/references" \
    "${DATA_DIR:-/workspace/data}/state" \
    "${HF_HOME:-/workspace/hf_cache}" \
    /var/log/novelconverter

echo "[entrypoint] Data dir: ${DATA_DIR:-/workspace/data}"
echo "[entrypoint] HF cache: ${HF_HOME:-/workspace/hf_cache}"
echo "[entrypoint] TTS workers: BASE_USE_REAL=${TTS_BASE_USE_REAL:-false} CUSTOM_USE_REAL=${TTS_CUSTOM_USE_REAL:-false} DESIGN_USE_REAL=${TTS_DESIGN_USE_REAL:-false}"
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="${PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK:-True}"
ensure_llama_server_ready || true

python3 - <<'PYDIAG'
import shutil
import sys

for mod in ("qwen_tts", "torch", "transformers", "torchaudio"):
    try:
        __import__(mod)
        print(f"[entrypoint] [ok] import {mod}")
    except Exception as exc:
        print(f"[entrypoint] [warning] import {mod} failed: {exc}", file=sys.stderr)

sox_path = shutil.which("sox")
if sox_path:
    print(f"[entrypoint] [ok] sox found: {sox_path}")
else:
    print("[entrypoint] [warning] sox not found in PATH", file=sys.stderr)
PYDIAG

echo "[entrypoint] Running prepare_assets..."
if ! python3 "${APP_DIR:-/workspace/NovelConverter}/docker/prepare_assets.py"; then
    echo "[entrypoint] ERROR: prepare_assets failed. Aborting startup."
    exit 1
fi

echo "[entrypoint] prepare_assets succeeded. Starting runtime services..."
exec "${APP_DIR:-/workspace/NovelConverter}/docker/start_services.sh" "$@"
