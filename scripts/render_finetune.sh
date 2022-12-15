#!/bin/bash

EXP_NAME=${1:-"train_DTU_2L_32H"}
SCAN_NUM=${2:-23}
POSE=${3:-"0"}
CUDA_VISIBLE_DEVICE=${4:-"0"}

shift
shift
shift
shift

echo "Rendering scan $SCAN_NUM pose $POSE for $EXP_NAME on GPU $CUDA_VISIBLE_DEVICE with flags $@"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICE python3 generator.py --config configs/render_experiment.txt --expname "render_${EXP_NAME}" --generate_specific_samples scan${SCAN_NUM} --gen_pose $POSE --no_parallel --ckpt_expname ${EXP_NAME} $@
