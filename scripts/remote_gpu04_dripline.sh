#!/bin/bash
# Remote GPU04 drip-line chains (2026-10-30 user decisions): eight one-GPU lanes,
# per-GPU budget 24 GiB, host <= 8 GiB per job, process guard, even-even nuclei.
# Each chain uses one canonical space, so S2n differences are consistent.
# Start: setsid nohup scripts/remote_gpu04_dripline.sh > dripline.log 2>&1 < /dev/null &
cd "$HOME/rhfb-nn/blocked-rhfb" || exit 1
export RHFB_AUTHORIZED_GPUS=0,1,2,3,4,5,6,7
PY=$HOME/rhfb-nn/env/bin/python
R=outputs/rhfb_spherical; D=$R/dripline; E=$R/dripline_eval
mkdir -p $D $E outputs/execution
STATUS=$R/dripline.status
GPU_MIB=24576

guard() { # gpu report timeout -- cmd...
  local gpu=$1 report=$2 timeout=$3; shift 4
  $PY scripts/run_rhfb_procguard.py --memory-mib 8192 --gpu-indices "$gpu" --gpu-memory-mib $GPU_MIB \
    --timeout-seconds "$timeout" --report "$report" -- "$@"
}
# one nucleus: gpu name z n epochs capn capp kneg kpos init stream
nucleus() {
  local gpu=$1 name=$2 z=$3 n=$4 epochs=$5 cn=$6 cp=$7 kn=$8 kp=$9 init=${10} stream=${11}
  local common="--z $z --n $n --capacity-n $cn --capacity-p $cp --kappa-limits $kn $kp $stream --device cuda --cuda-budget-mib $GPU_MIB"
  if [ ! -f $D/$name/summary.json ]; then
    local resume=""; [ -f $D/$name/training_checkpoint.pt ] && resume="--resume"
    local initarg=""; [ -n "$init" ] && initarg="--initial-model $init"
    guard $gpu outputs/execution/drip_train_$name.json 604800 -- /usr/bin/env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      $PY -u -m dpl_rhf.cli.rhfb_spherical_train $common --epochs $epochs --lr 2e-3 --logit-lr 5e-2 \
      $initarg $resume --out $D/$name >> $D/$name.log 2>&1
    echo "$name trained [gpu $gpu] $(grep -o '"status": "[a-z_]*"' outputs/execution/drip_train_$name.json) $(date)" >> $STATUS
  fi
  [ -f $D/$name/training_checkpoint.pt ] || return 1
  # Already evaluated after its final training: do not recompute on restart.
  if [ -f $E/$name/summary.json ] && [ $E/$name/summary.json -nt $D/$name/summary.json ]; then
    return 0
  fi
  mkdir -p $E/$name; cp $D/$name/history.csv $E/$name/ 2>/dev/null
  guard $gpu outputs/execution/drip_eval_$name.json 14400 -- /usr/bin/env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY -u -m dpl_rhf.cli.rhfb_spherical_train $common \
    --evaluate-only $D/$name/training_checkpoint.pt --evaluate-state last --out $E/$name > $E/$name.log 2>&1
  echo "$name evaluated [gpu $gpu] $(grep -o '"status": "[a-z_]*"' outputs/execution/drip_eval_$name.json) $(date)" >> $STATUS
}
# chain: gpu element z first_n last_n capn capp kneg kpos first_init first_epochs stream
chain() {
  local gpu=$1 el=$2 z=$3 n0=$4 n1=$5 cn=$6 cp=$7 kn=$8 kp=$9 init=${10} epochs0=${11} stream=${12}
  local n=$n0 prev="$init" epochs=$epochs0
  while [ $n -le $n1 ]; do
    local name=$el$((z+n))
    nucleus $gpu $name $z $n $epochs $cn $cp $kn $kp "$prev" "$stream" || { echo "$name FAILED, lane stops" >> $STATUS; return; }
    prev=$D/$name/model_final.pt; epochs=600; n=$((n+2))
  done
  echo "lane $el N=$n0..$n1 [gpu $gpu] done $(date)" >> $STATUS
}
# 2026-10-31: batched Fock (dpl_rhf/rhf/fock_batched.py) replaces per-pair
# streaming; Pb248 needs ~8.3 GiB and runs ~29x faster per epoch on A100.
ONE=""
echo "dripline start $(date)" >> $STATUS
chain 0 O  8  12  22  40  28 5 4 $R/production/O18/model_final.pt 600 ""   & sleep 20
chain 1 Sn 50 82  96  184 58 8 7 "" 1500 "$ONE" & sleep 20
chain 2 Sn 50 98  112 184 58 8 7 "" 1500 "$ONE" & sleep 20
chain 3 Sn 50 114 130 184 58 8 7 "" 1500 "$ONE" & sleep 20
chain 4 Pb 82 126 140 240 92 8 7 "" 1500 "$ONE" & sleep 20
chain 5 Pb 82 142 156 240 92 8 7 "" 1500 "$ONE" & sleep 20
chain 6 Pb 82 158 172 240 92 8 7 "" 1500 "$ONE" & sleep 20
chain 7 Pb 82 174 188 240 92 8 7 "" 1500 "$ONE" &
wait
echo DRIPLINE_DONE >> $STATUS
