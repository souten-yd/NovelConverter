#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="$REPO_ROOT/.venv_orchestrator"

echo "=== Setting up orchestrator venv ==="
python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install -r "$REPO_ROOT/app/orchestrator/requirements.txt"

echo ""
echo "✓ Orchestrator venv ready at $VENV"
echo "  Activate: source $VENV/bin/activate"
