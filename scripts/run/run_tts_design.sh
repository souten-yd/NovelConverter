#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="$REPO_ROOT/.venv_tts_design"
PORT="${TTS_DESIGN_PORT:-8003}"

if [ ! -d "$VENV" ]; then
  echo "ERROR: venv not found. Run scripts/setup/setup_tts_design.sh first."
  exit 1
fi

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export TTS_DESIGN_PORT="$PORT"
export TTS_DESIGN_USE_REAL="${TTS_DESIGN_USE_REAL:-false}"

echo "Starting TTS Design Worker on port $PORT (mock=${TTS_DESIGN_USE_REAL}) ..."
"$VENV/bin/uvicorn" app.workers.tts_design.main:app --host 0.0.0.0 --port "$PORT"
