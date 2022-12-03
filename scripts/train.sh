#!/bin/bash

git pull
echo "" > nohup.out

NUM_PROC=$(nproc)
echo "NOTE: Using $NUM_PROC workers (number of CPUs in VM)."

python trainer.py --config configs/train_DTU.txt --num_workers=$NUM_PROC $@