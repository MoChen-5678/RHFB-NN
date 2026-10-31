#!/usr/bin/env bash
set -euo pipefail
if (($# != 1)); then
  echo "Usage: $0 /absolute/path/to/Core-1204" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "$ROOT/compare_with_core.sh" \
  "$ROOT/work/outputs/PKA1_O16" 8 8 16 "$1" \
  "$ROOT/work/outputs/PKA1_O16/core_comparison"
