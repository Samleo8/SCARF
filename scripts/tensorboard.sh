#!/bin/bash

# This script is used to start tensorboard in a headless environment
EXP_NAME=${1:-"train_DTU"}

LOG_DIR=logs/${EXP_NAME}/tensorboard
if [ $1 == "all" ]; then
    LOG_DIR=logs
fi

killall tensorboard
nohup tensorboard --logdir=LOG_DIR --port=6006 &>/dev/null &

PID=$!
echo "Tensorboard started in background (${PID}) at port 6006"
echo ${PID} >.tb_pid
