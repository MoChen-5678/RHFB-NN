#!/bin/bash
# Remote GPU04: Fortran S-RHFB-C KMX=(8,8) references along the drip-line chains
# (CPU only, process guard <= 16 GiB, one Fortran job at a time).
cd "$HOME/rhfb-nn/blocked-rhfb" || exit 1
PY=$HOME/rhfb-nn/env/bin/python
EXE=$HOME/rhfb-nn/fortran/S-RHFB-C-kmx88/rhfb.exe
RUNS=$HOME/rhfb-nn/fortran/runs_kmx88
STATUS=$RUNS/chains.status
one() { # name z n
  local d=$RUNS/$1
  [ -f $d/parsed.json ] && return 0
  $PY scripts/fortran_rhfb_reference.py write-input --nucleus $1 --z $2 --n $3 --out $d > /dev/null
  cp $EXE $d/rhfb.exe
  $PY scripts/run_rhfb_procguard.py --memory-mib 16384 --threads 8 --timeout-seconds 7200 \
    --report $d/guard.json -- bash -c "cd $d && ./rhfb.exe > stdout.log 2>&1"
  $PY scripts/fortran_rhfb_reference.py parse --run-dir $d --out $d/parsed.json > $d/parse.log 2>&1
  echo "$1 $(grep -o '"status": "[a-z_]*"' $d/guard.json) $(date)" >> $STATUS
}
for n in $(seq 12 2 22); do one O$((8+n)) 8 $n; done
for n in $(seq 82 2 130); do one Sn$((50+n)) 50 $n; done
for n in $(seq 126 2 188); do one Pb$((82+n)) 82 $n; done
echo FORTRAN_CHAINS_DONE >> $STATUS
