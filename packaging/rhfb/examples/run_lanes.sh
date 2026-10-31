#!/bin/bash
# Template: independent lanes of two GPUs each, run in parallel (one nucleus per lane at a time).
# Edit the lanes to the cards you are allowed to use; export RHFB_AUTHORIZED_GPUS accordingly.
# Start detached so a closed SSH session does not stop it:
#   setsid nohup examples/run_lanes.sh > logs/lanes.out 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
export RHFB_AUTHORIZED_GPUS=${RHFB_AUTHORIZED_GPUS:-0,1,2,3,4,5,6,7}
mkdir -p logs
lane1() { examples/run_one.sh O16  8  8   500 40  28 5 4 0,1
          examples/run_one.sh O18  8  10  500 40  28 5 4 0,1 O16
          examples/run_one.sh O17  8  9   500 40  28 5 4 0,1 O16 -3; }
lane2() { examples/run_one.sh Ca40 20 20  500 50  40 5 4 2,3
          examples/run_one.sh Ca48 20 28  500 50  40 5 4 2,3 Ca40; }
lane3() { examples/run_one.sh Sn132 50 82 500 100 58 7 6 4,5; }
lane4() { examples/run_one.sh Pb208 82 126 500 148 92 8 7 6,7
          examples/run_one.sh Pb210 82 128 500 148 92 8 7 6,7 Pb208; }
lane1 & sleep 20; lane2 & sleep 20; lane3 & sleep 20; lane4 &
wait
echo LANES_DONE >> logs/status
