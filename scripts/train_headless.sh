#!/bin/bash

git pull
killall python3 # kill all previous running processes
echo "" > nohup.out

nohup ./scripts/train.sh &
echo $! > .train_pid