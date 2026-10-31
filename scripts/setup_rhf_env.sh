#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${DPL_RHF_ENV_PREFIX:-$ROOT/.dpl-rhf-env}"
MODE="auto"
TORCH_INDEX_URL="${DPL_RHF_TORCH_INDEX_URL:-}"
PYTHON_VERSION="${DPL_RHF_PYTHON_VERSION:-3.10}"

usage() {
  cat <<'EOF'
Usage: ./setup_env.sh [--cuda|--cpu] [--env-prefix PATH] [--torch-index-url URL]

Creates an isolated Conda environment, installs PyTorch and DPL-RHF, and runs
the float64/second-derivative/disk-offload self-check. If Conda is unavailable,
Miniforge is installed locally inside the extracted release.
EOF
}

while (($#)); do
  case "$1" in
    --cuda) MODE="cuda" ;;
    --cpu) MODE="cpu" ;;
    --env-prefix) shift; ENV_PREFIX="${1:?missing environment path}" ;;
    --torch-index-url) shift; TORCH_INDEX_URL="${1:?missing index URL}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if [[ "$MODE" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    MODE="cuda"
  else
    MODE="cpu"
  fi
fi

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
else
  ARCH="$(uname -m)"
  case "$ARCH" in
    x86_64|amd64) INSTALLER_ARCH="x86_64" ;;
    aarch64|arm64) INSTALLER_ARCH="aarch64" ;;
    *) echo "Unsupported Linux architecture: $ARCH" >&2; exit 1 ;;
  esac
  CONDA_BASE="$ROOT/.miniforge3"
  if [[ ! -x "$CONDA_BASE/bin/conda" ]]; then
    INSTALLER="$ROOT/.miniforge-installer.sh"
    URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${INSTALLER_ARCH}.sh"
    echo "Conda not found; downloading Miniforge from $URL"
    if command -v curl >/dev/null 2>&1; then
      curl -fL "$URL" -o "$INSTALLER"
    elif command -v wget >/dev/null 2>&1; then
      wget -O "$INSTALLER" "$URL"
    else
      echo "curl or wget is required to bootstrap Miniforge" >&2
      exit 1
    fi
    bash "$INSTALLER" -b -p "$CONDA_BASE"
    rm -f "$INSTALLER"
  fi
fi

# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  conda create -y -p "$ENV_PREFIX" "python=$PYTHON_VERSION" pip setuptools wheel
fi
conda activate "$ENV_PREFIX"

python -m pip install --upgrade pip setuptools wheel
python -m pip install "numpy>=1.24,<3" "scipy>=1.10" "matplotlib>=3.7" "sympy>=1.12"

if [[ -n "$TORCH_INDEX_URL" ]]; then
  python -m pip install "torch==2.5.1" --index-url "$TORCH_INDEX_URL"
elif [[ "$MODE" == "cuda" && "$(uname -m)" == "x86_64" ]]; then
  python -m pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121
elif [[ "$MODE" == "cpu" && "$(uname -m)" == "x86_64" ]]; then
  python -m pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cpu
else
  python -m pip install "torch==2.5.1"
fi

python -m pip install -e "$ROOT" --no-deps --no-build-isolation
printf '%s\n' "$ENV_PREFIX" > "$ROOT/.dpl-rhf-env-path"

CHECK_ARGS=()
if [[ "$MODE" == "cuda" ]]; then CHECK_ARGS+=(--require-cuda); fi
python "$ROOT/scripts/check_rhf_environment.py" "${CHECK_ARGS[@]}"

echo
echo "DPL-RHF environment is ready."
echo "Activate it with: source \"$ROOT/activate_env.sh\""
echo "Run an example with: ./run_rhf.sh PKA1 8 8 16 5 memory"
