#!/bin/bash

EXP_NAME=${1:-"train_DTU_2L_16H"}
SAMPLE=${2:-"scan23"}
CUDA_VISIBLE_DEVICE=${3:-"0"}

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICE python3 generator.py --config configs/render_experiment.txt --expname render_experiment_${SAMPLE} --generate_specific_samples ${SAMPLE} --gen_pose 0 --no_parallel --ckpt_expname ${EXP_NAME}
