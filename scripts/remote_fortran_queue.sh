#!/bin/bash
# Remote GPU04: Fortran S-RHFB-C KMX=(8,8) references, CPU only, process guard (<=16 GiB).
cd "$HOME/rhfb-nn/blocked-rhfb" || exit 1
export RHFB_AUTHORIZED_GPUS=0,1,2,3,4,5,6,7  # GPU04, 2026-10-28 user decision
PY=$HOME/rhfb-nn/env/bin/python
EXE=$HOME/rhfb-nn/fortran/S-RHFB-C-kmx88/rhfb.exe
RUNS=$HOME/rhfb-nn/fortran/runs_kmx88
STATUS=$RUNS/status
mkdir -p $RUNS outputs/execution
for spec in "O16 8 8" "Ca40 20 20" "Ca48 20 28" "Sn132 50 82" "Pb208 82 126" "Pb210 82 128"; do
  set -- $spec
  d=$RUNS/$1
  [ -f $d/parsed.json ] && continue
  $PY scripts/fortran_rhfb_reference.py write-input --nucleus $1 --z $2 --n $3 --out $d > /dev/null
  cp $EXE $d/rhfb.exe
  $PY scripts/run_rhfb_procguard.py --memory-mib 16384 --threads 8 --timeout-seconds 43200 \
    --report $d/guard.json -- bash -c "cd $d && ./rhfb.exe > stdout.log 2>&1"
  $PY scripts/fortran_rhfb_reference.py parse --run-dir $d --out $d/parsed.json > $d/parse.log 2>&1
  echo "$1 $(grep -o '"status": "[a-z_]*"' $d/guard.json) peak_GB=$($PY -c "import json;print(round(json.load(open('$d/guard.json'))['peak_rss_bytes']/1e9,2))") $(date)" >> $STATUS
done
echo FORTRAN_QUEUE_DONE >> $STATUS
