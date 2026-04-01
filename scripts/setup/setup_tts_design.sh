#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="$REPO_ROOT/.venv_tts_design"

echo "=== Setting up TTS Design venv ==="
python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install -r "$REPO_ROOT/app/workers/tts_design/requirements.txt"

echo ""
echo "✓ TTS Design venv ready at $VENV"
