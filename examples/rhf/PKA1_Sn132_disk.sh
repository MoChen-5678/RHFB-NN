#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "$ROOT/run_background.sh" PKA1 50 82 132 500 disk "$ROOT/work/outputs/PKA1_Sn132"
