#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f "$ROOT/.dpl-rhf-env-path" ]]; then
  echo "Environment is not configured. Run ./setup_env.sh first." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$ROOT/activate_env.sh" >/dev/null

MODEL="${1:-PKA1}"
Z="${2:-8}"
N="${3:-8}"
A="${4:-$((Z + N))}"
EPOCHS="${5:-500}"
CACHE_MODE="${6:-memory}"
OUT="${7:-$ROOT/work/outputs/${MODEL}_Z${Z}_N${N}}"
if (($# >= 7)); then shift 7; else set --; fi

case "$CACHE_MODE" in
  memory|disk|recompute) ;;
  *) echo "Cache mode must be memory, disk or recompute" >&2; exit 2 ;;
esac

mkdir -p "$OUT" "$ROOT/work/graph-cache"
exec python -m dpl_rhf.cli.rhf_train \
  --model "$MODEL" --z "$Z" --n "$N" --a "$A" \
  --epochs "$EPOCHS" --device auto --resume \
  --graph-offload "$CACHE_MODE" \
  --graph-offload-dir "$ROOT/work/graph-cache" \
  --graph-offload-min-mib 0.25 \
  --checkpoint-every 10 --refresh-interval 100 \
  --out "$OUT" "$@"
