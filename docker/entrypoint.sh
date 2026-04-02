#!/usr/bin/env bash
# =============================================================================
# NovelConverter – Docker entrypoint
# - Creates runtime directories
# - Waits for workers to be healthy before reporting ready
# - Starts supervisord (manages all 4 services)
# =============================================================================
set -euo pipefail

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
