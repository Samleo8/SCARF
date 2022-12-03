#!/bin/bash

git pull
killall python3 # kill all previous running processes
echo "" > nohup.out

nohup ./scripts/train.sh $@ &

PID=$!
echo "Training started in background task ${PID}. To see the output, run 'tail -f nohup.out'"
echo ${PID} > .train_pid