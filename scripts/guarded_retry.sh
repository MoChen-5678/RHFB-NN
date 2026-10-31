#!/bin/bash
# Run one numerical job through run_rhfb_limited.py, waiting while the shared
# lock is held or the authorized GPU is busy. Never runs a job unguarded.
#   guarded_retry.sh [--gpu] [--long] REPORT MEM_MIB TIMEOUT_S -- CMD...
# --gpu resolves physical GPU indices (GPU_INDICES, default 1) to current UUIDs at every attempt.
cd /home/ubuntu/rhf/DFT-RHF-NoRefresh/variants/blocked-rhfb || exit 1
PY=/home/ubuntu/anaconda3/envs/torch_env/bin/python
gpu=0; long=""
while [ "${1:0:2}" = "--" ] && [ "$1" != "--" ]; do
  case $1 in --gpu) gpu=1 ;; --long) long=--long-run ;; esac; shift
done
report=$1; mem=$2; timeout=$3; shift 4
for attempt in $(seq 1 2200); do
  args=()
  if [ $gpu -eq 1 ]; then
    # GPU_INDICES (default "1") lists physical indices, primary first; each is
    # resolved to its current UUID at every attempt.
    uuids=""
    for index in ${GPU_INDICES:-1}; do
      uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v i=$index '$1==i{print $2}')
      [ -n "$uuid" ] || { echo "physical GPU $index not found"; exit 98; }
      uuids="${uuids:+$uuids,}$uuid"
    done
    args=(--gpu-uuid "$uuids" --gpu-memory-mib "${GPU_MIB:-8192}")
  fi
  out=$($PY scripts/run_rhfb_limited.py $long --memory-mib "$mem" --timeout-seconds "$timeout" \
        "${args[@]}" --report "$report" -- "$@" 2>&1); code=$?
  if echo "$out" | grep -q "concurrent runs are refused\|still active; numerical launch refused\|execution lock\|unavailable or busy\|existing compute process"; then
    sleep 20; continue
  fi
  echo "$out"
  echo "GUARD_STATUS=$($PY -c "import json;print(json.load(open('$report'))['status'])" 2>/dev/null)"
  exit $code
done
echo "lock never freed"; exit 99
