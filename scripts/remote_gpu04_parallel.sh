#!/bin/bash
# Remote GPU04, all eight GPUs (2026-10-28 user decision): four parallel lanes of
# two GPUs each. Every job runs under the process guard (host <= 8 GiB, per-GPU
# <= 8 GiB, allocator limit, per-GPU locks). Tests already passed on this host.
cd "$HOME/rhfb-nn/blocked-rhfb" || exit 1
export RHFB_AUTHORIZED_GPUS=0,1,2,3,4,5,6,7  # GPU04, 2026-10-28 user decision
PY=$HOME/rhfb-nn/env/bin/python
R=outputs/rhfb_spherical; P=$R/production; E=$R/production_eval
mkdir -p $P $E outputs/execution
STATUS=$R/remote_parallel.status
STREAM="--fock-stream-devices cuda:0,cuda:1"

guard() { # gpus report timeout -- cmd...
  local gpus=$1 report=$2 timeout=$3; shift 4
  $PY scripts/run_rhfb_procguard.py --memory-mib 8192 --gpu-indices "$gpus" --gpu-memory-mib 8192 \
    --timeout-seconds "$timeout" --report "$report" -- "$@"
}
train() { # gpus name | after shift: z n epochs capn capp klneg klpos init block
  local gpus=$1 name=$2 resume=""; shift 2
  [ -f $P/$name/summary.json ] && return 0
  [ -f $P/$name/training_checkpoint.pt ] && resume="--resume"
  guard $gpus outputs/execution/remote_prod_$name.json 172800 -- /usr/bin/env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY -u -m dpl_rhf.cli.rhfb_spherical_train --z $1 --n $2 --epochs $3 \
    --capacity-n $4 --capacity-p $5 --kappa-limits $6 $7 --lr 2e-3 --logit-lr 5e-2 $STREAM \
    ${8:+--initial-model $P/$8/model_final.pt} ${9:+--block-n $9} $resume \
    --device cuda --cuda-budget-mib 8192 --out $P/$name >> $P/$name.log 2>&1
  echo "$name trained [$gpus] $(grep -o '"status": "[a-z_]*"' outputs/execution/remote_prod_$name.json) $(date)" >> $STATUS
}
reeval() { # gpus name | after shift: z n capn capp klneg klpos block
  local gpus=$1 name=$2; shift 2
  mkdir -p $E/$name; cp $P/$name/history.csv $E/$name/ 2>/dev/null
  guard $gpus outputs/execution/remote_reeval_$name.json 7200 -- /usr/bin/env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY -u -m dpl_rhf.cli.rhfb_spherical_train --z $1 --n $2 --capacity-n $3 --capacity-p $4 \
    --kappa-limits $5 $6 ${7:+--block-n $7} $STREAM --evaluate-only $P/$name/training_checkpoint.pt \
    --evaluate-state last --device cuda --cuda-budget-mib 8192 --out $E/$name > $E/$name.log 2>&1
  echo "$name reevaluated [$gpus] $(grep -o '"status": "[a-z_]*"' outputs/execution/remote_reeval_$name.json) $(date)" >> $STATUS
}

lane1() {
  train  0,1 Ca48  20 28  500  50  40 5 4 Ca40 ""
  reeval 0,1 Ca48  20 28  50  40 5 4 ""
  reeval 0,1 O17   8  9   40  28 5 4 -3
  reeval 0,1 O18   8  10  40  28 5 4 ""
}
lane2() {
  train  2,3 Sn132 50 82  500 100 58 7 6 "" ""
  reeval 2,3 Sn132 50 82  100 58 7 6 ""
}
lane3() {
  train  4,5 Pb208 82 126 500 148 92 8 7 "" ""
  reeval 4,5 Pb208 82 126 148 92 8 7 ""
}
lane4() {
  # Started in parallel from an oscillator start rather than waiting for Pb208.
  train  6,7 Pb210 82 128 500 148 92 8 7 "" ""
  reeval 6,7 Pb210 82 128 148 92 8 7 ""
}
echo "parallel start $(date)" >> $STATUS
lane1 & sleep 30
lane2 & sleep 30
lane3 & sleep 30
lane4 &
wait
echo REMOTE_PARALLEL_DONE >> $STATUS
