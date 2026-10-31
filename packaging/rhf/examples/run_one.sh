#!/bin/bash
# Train one nucleus with the spherical no-refresh neural RHF solver (PKA1 or PKO2)
# under the process guard.  Settings are those of the validated O16/Ca40/Ca48 runs.
#
# usage: examples/run_one.sh NAME Z N EPOCHS GPUS [MODEL]
#   GPUS   physical indices, e.g. "0" (one card) or "0,1" (Fock exchange streamed over two cards)
#   MODEL  PKA1 (default) or PKO2
# env:  RHFB_AUTHORIZED_GPUS (required for GPU jobs)  HOST_MIB (8192)  GPU_MIB (8192)  THREADS (2)
#       EXTRA="..." appended to the training command (e.g. "--graph-offload recompute")
set -u
cd "$(dirname "$0")/.."
PY=$PWD/env/bin/python
NAME=$1 Z=$2 N=$3 EPOCHS=$4 GPUS=$5 MODEL=${6:-PKA1}
mkdir -p runs logs
NDEV=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
STREAM=""; [ "$NDEV" -ge 2 ] && STREAM="--fock-stream-devices $(seq -s, -f 'cuda:%g' 0 $((NDEV-1)))"
RESUME=""; [ -f runs/$NAME/training_checkpoint.pt ] && RESUME="--resume"
$PY scripts/run_rhfb_procguard.py --memory-mib "${HOST_MIB:-8192}" --gpu-indices "$GPUS" \
  --gpu-memory-mib "${GPU_MIB:-8192}" --threads "${THREADS:-2}" --timeout-seconds 1209600 \
  --report logs/guard_$NAME.json -- /usr/bin/env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY -u -m dpl_rhf.cli.rhf_train --model $MODEL --z $Z --n $N --epochs $EPOCHS \
  --energy-weight 2.0 --stationarity-weight 1.0 --inverse-weight 10.0 --lr 1e-3 \
  --device cuda $STREAM $RESUME ${EXTRA:-} --out runs/$NAME >> logs/$NAME.log 2>&1
echo "$NAME: $(grep -o '"status": "[a-z_]*"' logs/guard_$NAME.json) $(date)" >> logs/status
