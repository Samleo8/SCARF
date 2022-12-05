#!/bin/bash

# This script is used to start tensorboard in a headless environment
EXP_NAME=${1:-"train_DTU"}

LOG_DIR="./logs/${EXP_NAME}/tensorboard"

shift
while [ $# -gt 0 ]; do
    EXP_NAME=$1
    LOG_DIR="${LOG_DIR},./logs/${EXP_NAME}/tensorboard"
    shift
done

# LOG_DIR="./logs/${EXP_NAME}/tensorboard"

killall tensorboard
nohup tensorboard --logdir_spec=$LOG_DIR --port=6006 &>/dev/null &

PID=$!
echo "Tensorboard started in background (${PID}) at port 6006"
echo ${PID} >.tb_pid
