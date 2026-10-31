#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python -m pip install . --no-deps --no-build-isolation
python -m dpl_rhf.cli.check_environment "$@"
