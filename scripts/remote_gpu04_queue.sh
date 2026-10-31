#!/bin/bash
# Remote GPU04 queue (physical GPUs 6,7 only; process-level guard; one job at a time).
# Tests first; production starts only if every test passes.
cd "$HOME/rhfb-nn/blocked-rhfb" || exit 1
export RHFB_AUTHORIZED_GPUS=0,1,2,3,4,5,6,7  # GPU04, 2026-10-28 user decision
PY=$HOME/rhfb-nn/env/bin/python
R=outputs/rhfb_spherical; P=$R/production; E=$R/production_eval
mkdir -p $P $E outputs/execution
STATUS=$R/remote_queue.status
STREAM="--fock-stream-devices cuda:0,cuda:1"
guard() { # report timeout -- cmd...
  local report=$1 timeout=$2; shift 3
  $PY scripts/run_rhfb_procguard.py --memory-mib 8192 --gpu-indices 6,7 --gpu-memory-mib 8192 \
    --threads "${THREADS:-2}" --timeout-seconds "$timeout" --report "$report" -- "$@"
}

# GPU-relevant tests only: the CPU m-scheme pairing oracle (test_spherical_d1s_pairing)
# was validated locally on 2026-10-26 with identical code and exceeds the 1 h budget here.
THREADS=8 guard outputs/execution/remote_tests.json 10800 -- /usr/bin/env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY -m pytest -q -p no:cacheprovider dpl_rhf/tests/test_fock_streaming.py \
  dpl_rhf/tests/test_spherical_rhfb_backend.py \
  > $R/remote_tests.log 2>&1
if ! grep -q '"status": "completed"' outputs/execution/remote_tests.json; then
  echo "TESTS_FAILED $(date)" >> $STATUS; exit 1
fi
echo "tests passed $(date)" >> $STATUS

run() { # name z n epochs capn capp klneg klpos init
  local name=$1 resume=""
  if [ ! -f $P/$name/summary.json ]; then
    [ -f $P/$name/training_checkpoint.pt ] && resume="--resume"
    guard outputs/execution/remote_prod_$name.json 172800 -- /usr/bin/env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      $PY -u -m dpl_rhf.cli.rhfb_spherical_train --z $2 --n $3 --epochs $4 \
      --capacity-n $5 --capacity-p $6 --kappa-limits $7 $8 --lr 2e-3 --logit-lr 5e-2 $STREAM \
      ${9:+--initial-model $P/$9/model_final.pt} $resume \
      --device cuda --cuda-budget-mib 8192 --out $P/$name >> $P/$name.log 2>&1
    echo "$name trained $(grep -o '"status": "[a-z_]*"' outputs/execution/remote_prod_$name.json) $(date)" >> $STATUS
  fi
  mkdir -p $E/$name; cp $P/$name/history.csv $E/$name/ 2>/dev/null
  guard outputs/execution/remote_reeval_$name.json 7200 -- /usr/bin/env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY -u -m dpl_rhf.cli.rhfb_spherical_train --z $2 --n $3 --capacity-n $5 --capacity-p $6 \
    --kappa-limits $7 $8 $STREAM --evaluate-only $P/$name/training_checkpoint.pt --evaluate-state last \
    --device cuda --cuda-budget-mib 8192 --out $E/$name > $E/$name.log 2>&1
  echo "$name reevaluated $(grep -o '"status": "[a-z_]*"' outputs/execution/remote_reeval_$name.json) $(date)" >> $STATUS
}
reeval_only() { # name z n capn capp klneg klpos block
  mkdir -p $E/$1; cp $P/$1/history.csv $E/$1/ 2>/dev/null
  guard outputs/execution/remote_reeval_$1.json 7200 -- /usr/bin/env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY -u -m dpl_rhf.cli.rhfb_spherical_train --z $2 --n $3 --capacity-n $4 --capacity-p $5 \
    --kappa-limits $6 $7 ${8:+--block-n $8} $STREAM --evaluate-only $P/$1/training_checkpoint.pt \
    --evaluate-state last --device cuda --cuda-budget-mib 8192 --out $E/$1 > $E/$1.log 2>&1
  echo "$1 reevaluated $(grep -o '"status": "[a-z_]*"' outputs/execution/remote_reeval_$1.json) $(date)" >> $STATUS
}
# Ca48 resumes its checkpoint (warm start from Ca40 already applied in that checkpoint).
run Ca48  20 28  800  50  40 5 4 Ca40
run Sn132 50 82  1500 100 58 7 6 ""
run Pb208 82 126 1500 148 92 8 7 ""
run Pb210 82 128 800  148 92 8 7 Pb208
reeval_only O17 8 9  40 28 5 4 -3
reeval_only O18 8 10 40 28 5 4 ""
echo REMOTE_QUEUE_DONE >> $STATUS
