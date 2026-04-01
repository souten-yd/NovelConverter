#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="$REPO_ROOT/.venv_orchestrator"
PORT="${ORCHESTRATOR_PORT:-8000}"

if [ ! -d "$VENV" ]; then
  echo "ERROR: venv not found. Run scripts/setup/setup_orchestrator.sh first."
  exit 1
fi

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export DATABASE_URL="${DATABASE_URL:-sqlite:///$REPO_ROOT/data/novelconverter.db}"

echo "Starting Orchestrator on port $PORT ..."
"$VENV/bin/uvicorn" app.orchestrator.main:app --host 0.0.0.0 --port "$PORT" --reload
