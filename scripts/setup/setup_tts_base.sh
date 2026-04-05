#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="$REPO_ROOT/.venv_tts_base"
PYTHON="$VENV/bin/python"

echo "=== Setting up TTS Base venv ==="
python3 -m venv "$VENV"
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -r "$REPO_ROOT/app/workers/tts_base/requirements.txt"

# Also need shared app package (editable install)
"$PYTHON" -m pip install -e "$REPO_ROOT" 2>/dev/null || \
  PYTHONPATH="$REPO_ROOT" echo "Note: install app as package if needed"
"$PYTHON" -c "import qwen_tts, torch, transformers; print('TTS base import preflight: ok')"

echo ""
echo "✓ TTS Base venv ready at $VENV"
