#!/bin/bash

# This script is used to start tensorboard in a headless environment
killall tensorboard
nohup tensorboard --logdir=logs/train_DTU/tensorboard --port=6006 &>/dev/null &

PID=$!
echo "Tensorboard started in background (${PID}) at port 6006"
echo ${PID} >.tb_pid
