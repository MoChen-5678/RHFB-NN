#!/bin/bash
# Acceptance tests under the guard. With GPUS="0,1" the two-device streaming test runs too.
# usage: RHFB_AUTHORIZED_GPUS=0,1 examples/run_tests.sh [GPUS]     (empty GPUS -> CPU only)
cd "$(dirname "$0")/.."
PY=$PWD/env/bin/python; GPUS=${1:-}; mkdir -p logs
$PY scripts/run_rhfb_procguard.py --memory-mib 8192 ${GPUS:+--gpu-indices $GPUS --gpu-memory-mib 8192} \
  --threads 8 --timeout-seconds 10800 --report logs/guard_tests.json -- \
  $PY -m pytest -q -p no:cacheprovider dpl_rhf/tests/test_fock_streaming.py \
  dpl_rhf/tests/test_spherical_rhfb_backend.py ${FULL:+dpl_rhf/tests/test_spherical_d1s_pairing.py}
