#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="$REPO_ROOT/.venv_tts_base"
PYTHON="$VENV/bin/python"
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
echo "Worker python: $("$PYTHON" -c 'import sys; print(sys.executable)')"
if [ "${TTS_BASE_USE_REAL}" = "true" ]; then
  "$PYTHON" -m pip show qwen-tts >/dev/null 2>&1 || "$PYTHON" -m pip install qwen-tts
  "$PYTHON" -c "import qwen_tts, torch, transformers, torchaudio; print('runtime preflight: ok')"
fi
"$PYTHON" -m uvicorn app.workers.tts_base.main:app --host 0.0.0.0 --port "$PORT"
