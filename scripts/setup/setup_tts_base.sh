#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="$REPO_ROOT/.venv_tts_base"

echo "=== Setting up TTS Base venv ==="
python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install -r "$REPO_ROOT/app/workers/tts_base/requirements.txt"

# Also need shared app package (editable install)
"$VENV/bin/pip" install -e "$REPO_ROOT" 2>/dev/null || \
  PYTHONPATH="$REPO_ROOT" echo "Note: install app as package if needed"

echo ""
echo "✓ TTS Base venv ready at $VENV"
echo "  To use real Qwen3-TTS: uncomment torch/transformers in requirements.txt and re-run"
