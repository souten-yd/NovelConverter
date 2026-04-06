#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
  exec "$@"
fi

export SUPERVISOR_SOCKET="${SUPERVISOR_SOCKET:-/var/run/supervisor.sock}"
echo "[start_services] Supervisor socket: ${SUPERVISOR_SOCKET}"
echo "[start_services] Starting supervisord..."
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/novelconverter.conf
