#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$ROOT/work/run/rhf.pid"
if [[ ! -f "$PID_FILE" ]]; then
  echo "No RHF PID file found"
  exit 0
fi
PID="$(<"$PID_FILE")"
if ! kill -0 "$PID" 2>/dev/null; then
  echo "RHF process $PID is already stopped"
  rm -f "$PID_FILE"
  exit 0
fi
kill -INT "$PID"
for _ in $(seq 1 30); do
  if ! kill -0 "$PID" 2>/dev/null; then
    rm -f "$PID_FILE"
    echo "Stopped RHF process $PID"
    exit 0
  fi
  sleep 1
done
echo "Process did not stop after SIGINT; sending SIGTERM"
kill -TERM "$PID"
rm -f "$PID_FILE"
