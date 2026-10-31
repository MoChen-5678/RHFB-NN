#!/bin/bash
# Train one nucleus with the spherical neural RHFB solver, then re-evaluate it
# self-consistently.  Every step runs under the process guard.
#
# usage: examples/run_one.sh NAME Z N EPOCHS CAP_N CAP_P KNEG KPOS GPUS [INIT_NAME] [BLOCK_KAPPA]
#   GPUS        physical indices, e.g. "0,1" (two cards, Fock streamed) or "0" (one card)
#   INIT_NAME   optional warm start from runs/<INIT_NAME>/model_final.pt (same canonical space)
#   BLOCK_KAPPA optional kappa of the equal-filling blocked neutron level (odd N), e.g. -3 for 1d5/2
# env:  RHFB_AUTHORIZED_GPUS  (required for GPU jobs, e.g. 0,1,2,3)
#       HOST_MIB (default 8192)  GPU_MIB (default 8192)  THREADS (default 2)
set -u
cd "$(dirname "$0")/.."
PY=$PWD/env/bin/python
NAME=$1 Z=$2 N=$3 EPOCHS=$4 CAPN=$5 CAPP=$6 KNEG=$7 KPOS=$8 GPUS=$9 INIT=${10:-} BLOCK=${11:-}
RUNS=runs; EVAL=runs_eval; mkdir -p $RUNS $EVAL logs
NDEV=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
STREAM=""; [ "$NDEV" -ge 2 ] && STREAM="--fock-stream-devices $(seq -s, -f 'cuda:%g' 0 $((NDEV-1)))"
guard() { # report timeout -- cmd
  local report=$1 timeout=$2; shift 3
  $PY scripts/run_rhfb_procguard.py --memory-mib "${HOST_MIB:-8192}" --gpu-indices "$GPUS" \
    --gpu-memory-mib "${GPU_MIB:-8192}" --threads "${THREADS:-2}" \
    --timeout-seconds "$timeout" --report "$report" -- "$@"
}
COMMON="--z $Z --n $N --capacity-n $CAPN --capacity-p $CAPP --kappa-limits $KNEG $KPOS ${BLOCK:+--block-n $BLOCK} $STREAM --device cuda --cuda-budget-mib ${GPU_MIB:-8192}"
if [ ! -f $RUNS/$NAME/summary.json ]; then
  RESUME=""; [ -f $RUNS/$NAME/training_checkpoint.pt ] && RESUME="--resume"
  guard logs/guard_train_$NAME.json 1209600 -- /usr/bin/env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY -u -m dpl_rhf.cli.rhfb_spherical_train $COMMON --epochs $EPOCHS --lr 2e-3 --logit-lr 5e-2 \
    ${INIT:+--initial-model $RUNS/$INIT/model_final.pt} $RESUME --out $RUNS/$NAME >> logs/$NAME.train.log 2>&1
  echo "$NAME trained: $(grep -o '"status": "[a-z_]*"' logs/guard_train_$NAME.json) $(date)" >> logs/status
fi
mkdir -p $EVAL/$NAME; cp $RUNS/$NAME/history.csv $EVAL/$NAME/ 2>/dev/null
guard logs/guard_eval_$NAME.json 7200 -- /usr/bin/env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY -u -m dpl_rhf.cli.rhfb_spherical_train $COMMON \
  --evaluate-only $RUNS/$NAME/training_checkpoint.pt --evaluate-state last --out $EVAL/$NAME > logs/$NAME.eval.log 2>&1
echo "$NAME evaluated: $(grep -o '"status": "[a-z_]*"' logs/guard_eval_$NAME.json) $(date)" >> logs/status
