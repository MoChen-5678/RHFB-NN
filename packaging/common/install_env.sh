#!/bin/bash
# Create a self-contained Python 3.10 environment in ./env and install the package.
# Usage: bash install_env.sh            (uses conda/mamba/micromamba if found, else python3.10 -m venv)
#        CONDA_EXE=/path/to/conda bash install_env.sh
set -euo pipefail
cd "$(dirname "$0")"
ENV_DIR="$PWD/env"
if [ -x "$ENV_DIR/bin/python" ]; then echo "env already exists: $ENV_DIR"; else
  CONDA_EXE="${CONDA_EXE:-$(command -v micromamba || command -v mamba || command -v conda || true)}"
  if [ -n "$CONDA_EXE" ]; then
    # Keep the package cache inside this directory; never touch other environments.
    export CONDA_PKGS_DIRS="$PWD/.pkgs"
    "$CONDA_EXE" create -y -p "$ENV_DIR" -c conda-forge python=3.10
  elif command -v python3.10 >/dev/null; then
    python3.10 -m venv "$ENV_DIR"
  else
    echo "Need conda/mamba/micromamba or python3.10"; exit 1
  fi
fi
"$ENV_DIR/bin/python" -m pip install --upgrade pip
"$ENV_DIR/bin/python" -m pip install -r requirements.txt
"$ENV_DIR/bin/python" -m pip install -e . --no-deps
"$ENV_DIR/bin/python" - <<'PY'
import torch, numpy, scipy
print("ENV_OK torch", torch.__version__, "cuda", torch.version.cuda,
      "gpus", torch.cuda.device_count(), "numpy", numpy.__version__, "scipy", scipy.__version__)
PY
