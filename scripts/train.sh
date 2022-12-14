#!/bin/bash

EXP_NAME=${1:-train_DTU_2L_32H}
CUDA_VISIBLE_DEVICE=$2
NUM_PROC=$3

git pull
echo "" > nohup.out

if [ -z "$NUM_PROC" ]; then
    NUM_PROC=$(nproc)
    echo "NOTE: Using $NUM_PROC workers (number of CPUs in VM)."
fi

if [ -z "$CUDA_VISIBLE_DEVICE" ]; then
    echo "NOTE: Using GPU $CUDA_VISIBLE_DEVICE."
    export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICE
fi

python3 trainer.py --config configs/${EXP_NAME}.txt --num_workers=$NUM_PROC --no_parallel $@