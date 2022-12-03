#!/bin/bash

killall python3 # kill all previous running processes

git pull
echo "" > nohup.out

# NUM_PROC=$(nproc)
# echo "NOTE: Using $NUM_PROC workers (number of CPUs in VM)."
NUM_PROC=12

python3 trainer.py --config configs/train_DTU.txt --num_workers=$NUM_PROC $@