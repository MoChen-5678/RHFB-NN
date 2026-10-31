#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$ROOT/work/run/rhf.pid"
LOG_PATH_FILE="$ROOT/work/run/rhf.log-path"
PID="none"
STATUS="stopped"
if [[ -f "$PID_FILE" ]]; then PID="$(<"$PID_FILE")"; fi
if [[ "$PID" != "none" ]] && kill -0 "$PID" 2>/dev/null; then STATUS="running"; fi
echo "status=$STATUS pid=$PID"
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; fi
if command -v free >/dev/null 2>&1; then free -h; fi
df -h "$ROOT"
du -sh "$ROOT/work/graph-cache" 2>/dev/null || true
if [[ -f "$LOG_PATH_FILE" ]]; then
  LOG_FILE="$(<"$LOG_PATH_FILE")"
  echo "--- $LOG_FILE ---"
  tail -n 20 "$LOG_FILE" 2>/dev/null || true
fi
