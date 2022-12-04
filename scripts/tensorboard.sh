#!/bin/bash

# This script is used to start tensorboard in a headless environment
EXP_NAME=${1:-"train_DTU"}

killall tensorboard
nohup tensorboard --logdir=logs/${EXP_NAME}/tensorboard --port=6006 &>/dev/null &

PID=$!
echo "Tensorboard started in background (${PID}) at port 6006"
echo ${PID} >.tb_pid
