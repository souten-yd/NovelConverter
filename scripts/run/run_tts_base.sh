#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="$REPO_ROOT/.venv_tts_base"
PORT="${TTS_BASE_PORT:-8001}"

if [ ! -d "$VENV" ]; then
  echo "ERROR: venv not found. Run scripts/setup/setup_tts_base.sh first."
  exit 1
fi

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export TTS_BASE_PORT="$PORT"
export TTS_BASE_USE_REAL="${TTS_BASE_USE_REAL:-false}"

echo "Starting TTS Base Worker on port $PORT (mock=${TTS_BASE_USE_REAL}) ..."
"$VENV/bin/uvicorn" app.workers.tts_base.main:app --host 0.0.0.0 --port "$PORT"
