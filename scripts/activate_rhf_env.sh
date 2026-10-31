#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${DPL_RHF_ENV_PREFIX:-}"
if [[ -z "$ENV_PREFIX" && -f "$ROOT/.dpl-rhf-env-path" ]]; then
  ENV_PREFIX="$(<"$ROOT/.dpl-rhf-env-path")"
fi
ENV_PREFIX="${ENV_PREFIX:-$ROOT/.dpl-rhf-env}"

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
elif [[ -x "$ROOT/.miniforge3/bin/conda" ]]; then
  CONDA_BASE="$ROOT/.miniforge3"
else
  echo "Conda environment not found. Run ./setup_env.sh first." >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/work/matplotlib}"
mkdir -p "$MPLCONFIGDIR"
echo "Activated DPL-RHF environment: $ENV_PREFIX"
