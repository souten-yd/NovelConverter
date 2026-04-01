#!/usr/bin/env bash
# Start all services in background, then tail logs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOGS_DIR="$REPO_ROOT/data/logs"
mkdir -p "$LOGS_DIR"

cleanup() {
  echo "Stopping all services..."
  kill "$PID_BASE" "$PID_CUSTOM" "$PID_DESIGN" "$PID_ORCH" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Starting TTS Base Worker (port 8001)..."
bash "$REPO_ROOT/scripts/run/run_tts_base.sh" > "$LOGS_DIR/tts_base.log" 2>&1 &
PID_BASE=$!

echo "Starting TTS Custom Worker (port 8002)..."
bash "$REPO_ROOT/scripts/run/run_tts_custom.sh" > "$LOGS_DIR/tts_custom.log" 2>&1 &
PID_CUSTOM=$!

echo "Starting TTS Design Worker (port 8003)..."
bash "$REPO_ROOT/scripts/run/run_tts_design.sh" > "$LOGS_DIR/tts_design.log" 2>&1 &
PID_DESIGN=$!

sleep 2

echo "Starting Orchestrator (port 8000)..."
bash "$REPO_ROOT/scripts/run/run_orchestrator.sh" > "$LOGS_DIR/orchestrator.log" 2>&1 &
PID_ORCH=$!

echo ""
echo "========================================="
echo " All services started!"
echo "   Orchestrator UI: http://localhost:8000"
echo "   TTS Base:        http://localhost:8001"
echo "   TTS Custom:      http://localhost:8002"
echo "   TTS Design:      http://localhost:8003"
echo " Logs: $LOGS_DIR/"
echo "========================================="
echo " Press Ctrl+C to stop all services."
echo ""

wait "$PID_ORCH"
