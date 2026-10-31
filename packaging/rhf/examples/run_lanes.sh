#!/bin/bash
# Template: parallel lanes, one nucleus per lane at a time.  Start detached:
#   setsid nohup examples/run_lanes.sh > logs/lanes.out 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
export RHFB_AUTHORIZED_GPUS=${RHFB_AUTHORIZED_GPUS:-0,1,2,3,4,5,6,7}
mkdir -p logs
lane1() { examples/run_one.sh O16  8  8   300 0; examples/run_one.sh Ca40 20 20 300 1; }
lane2() { examples/run_one.sh Ca48 20 28  300 2,3; }
lane3() { examples/run_one.sh Sn132 50 82 300 4,5; }
lane4() { examples/run_one.sh Pb208 82 126 300 6,7; }
lane1 & sleep 20; lane2 & sleep 20; lane3 & sleep 20; lane4 &
wait
echo LANES_DONE >> logs/status
