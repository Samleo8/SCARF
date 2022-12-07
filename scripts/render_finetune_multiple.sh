#!/bin/bash

EXP_NAMES=(train_DTU train_DTU_2L_16H train_DTU_2L_nocompress)

CNT=0
for EXP_NAME in "${EXP_NAMES[@]}"; do
    echo "Processing $EXP_NAME on GPU $CNT"

    CUDA_VISIBLE_DEVICES=$CNT python3 generator.py --config configs/finetune_scan23.txt --generate_specific_samples scan23 --gen_pose 0 --no_parallel --ckpt_expname ${EXP_NAME}

    CNT=$((CNT + 1))
done