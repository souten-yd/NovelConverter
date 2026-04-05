#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="$REPO_ROOT/.venv_tts_custom"
PYTHON="$VENV/bin/python"

echo "=== Setting up TTS Custom venv ==="
python3 -m venv "$VENV"
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -r "$REPO_ROOT/app/workers/tts_custom/requirements.txt"
"$PYTHON" -m pip install -e "$REPO_ROOT"
"$PYTHON" -c "import qwen_tts, torch, transformers; print('TTS custom import preflight: ok')"

echo ""
echo "✓ TTS Custom venv ready at $VENV"
