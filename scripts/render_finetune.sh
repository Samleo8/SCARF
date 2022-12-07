#!/bin/bash

EXP_NAME=${1:-"train_DTU_2L_16H"}
python3 generator.py --config configs/finetune_scan23.txt --generate_specific_samples scan23 --gen_pose 0 --no_parallel --ckpt_expname ${EXP_NAME}