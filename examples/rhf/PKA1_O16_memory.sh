#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "$ROOT/run_rhf.sh" PKA1 8 8 16 500 memory "$ROOT/work/outputs/PKA1_O16"
