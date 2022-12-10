#!/bin/bash

git pull
echo "" >nohup.out

# kill all previous running processes
kill $(cat .train_pid)
for i in {1..5}; do
    killall python3 && echo "Successfully killed all python processes" || echo "No python processes to kill"
done

# run training headless
nohup ./scripts/train.sh $@ &

# save PID
PID=$!
echo "Training started in background task ${PID}. To see the output, run 'tail -f nohup.out'"
echo ${PID} >.train_pid

# check nohup
tail -f nohup.out
