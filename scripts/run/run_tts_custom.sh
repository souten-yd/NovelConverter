#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="$REPO_ROOT/.venv_tts_custom"
PORT="${TTS_CUSTOM_PORT:-8002}"

if [ ! -d "$VENV" ]; then
  echo "ERROR: venv not found. Run scripts/setup/setup_tts_custom.sh first."
  exit 1
fi

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export TTS_CUSTOM_PORT="$PORT"
export TTS_CUSTOM_USE_REAL="${TTS_CUSTOM_USE_REAL:-false}"

echo "Starting TTS Custom Worker on port $PORT (mock=${TTS_CUSTOM_USE_REAL}) ..."
"$VENV/bin/uvicorn" app.workers.tts_custom.main:app --host 0.0.0.0 --port "$PORT"
