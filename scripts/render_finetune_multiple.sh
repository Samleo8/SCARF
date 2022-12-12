#!/bin/bash

EXP_NAMES=(train_DTU_2L_32H train_DTU_4L_16H train_DTU_2L_16H_reduced)

CNT=1
PIDS=""
for EXP_NAME in "${EXP_NAMES[@]}"; do
    echo "Processing $EXP_NAME on GPU $CNT"

    CUDA_VISIBLE_DEVICES=$CNT python3 generator.py --config configs/finetune_scan23.txt --generate_specific_samples scan23 --gen_pose 0 --no_parallel --ckpt_expname ${EXP_NAME} &
    PIDS="$! $PIDS"

    CNT=$((CNT + 1))
done

wait $PIDS
echo "Done!"
