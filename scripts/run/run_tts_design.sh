#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="$REPO_ROOT/.venv_tts_design"
PYTHON="$VENV/bin/python"
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
echo "Worker python: $("$PYTHON" -c 'import sys; print(sys.executable)')"
if [ "${TTS_DESIGN_USE_REAL}" = "true" ]; then
  "$PYTHON" -m pip show qwen-tts >/dev/null 2>&1 || "$PYTHON" -m pip install qwen-tts
  "$PYTHON" -c "import qwen_tts, torch, transformers, torchaudio; print('runtime preflight: ok')"
fi
"$PYTHON" -m uvicorn app.workers.tts_design.main:app --host 0.0.0.0 --port "$PORT"
