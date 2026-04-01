#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="$REPO_ROOT/.venv_tts_custom"

echo "=== Setting up TTS Custom venv ==="
python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install -r "$REPO_ROOT/app/workers/tts_custom/requirements.txt"

echo ""
echo "✓ TTS Custom venv ready at $VENV"
