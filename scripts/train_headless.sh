#!/bin/bash

git pull

NUM=$1

echo "" >nohup${NUM}.out

# kill all previous running processes
if [[ -z $NUM ]]; then
    kill $(cat .train_pid)
    for i in {1..5}; do
        killall python3 && echo "Successfully killed all python processes" || echo "No python processes to kill"
    done
fi

# run training headless
nohup ./scripts/train${NUM}.sh $@ &

# save PID
PID=$!
echo "Training (train${NUM}) started in background task ${PID}. To see the output, run 'tail -f nohup${NUM}.out'"
echo ${PID} >.train${NUM}_pid

# check nohup
tail -f nohup${NUM}.out
