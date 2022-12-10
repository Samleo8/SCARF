#!/bin/bash

git pull
echo "" >nohup3.out

# run training headless
nohup ./scripts/train3.sh $@ &>nohup3.out &

# save PID
PID=$!
echo "Parallel training (train2) started in background task ${PID}. To see the output, run 'tail -f nohup3.out'"
echo ${PID} >.train3_pid

# check nohup
tail -f nohup3.out
