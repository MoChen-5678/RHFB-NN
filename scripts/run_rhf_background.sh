#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/work/logs" "$ROOT/work/run"
PID_FILE="$ROOT/work/run/rhf.pid"
LOG_FILE="$ROOT/work/logs/rhf-$(date +%Y%m%d-%H%M%S).log"

if [[ -f "$PID_FILE" ]] && kill -0 "$(<"$PID_FILE")" 2>/dev/null; then
  echo "An RHF process is already running with PID $(<"$PID_FILE")" >&2
  exit 1
fi

nohup "$ROOT/run_rhf.sh" "$@" >>"$LOG_FILE" 2>&1 </dev/null &
PID=$!
printf '%s\n' "$PID" > "$PID_FILE"
printf '%s\n' "$LOG_FILE" > "$ROOT/work/run/rhf.log-path"
echo "Started RHF PID $PID"
echo "Log: $LOG_FILE"
echo "Monitor: ./monitor_rhf.sh"
