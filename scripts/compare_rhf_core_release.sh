#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if (($# < 6)); then
  echo "Usage: ./compare_with_core.sh RESULT_DIR Z N A CORE_DIR OUT_DIR" >&2
  exit 2
fi
# shellcheck disable=SC1091
source "$ROOT/activate_env.sh" >/dev/null

exec python -m dpl_rhf.cli.compare_rhf_core1204 \
  --result-dir "$1" --z "$2" --n "$3" --a "$4" \
  --core-dir "$5" --out "$6"
