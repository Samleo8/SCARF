#!/bin/bash

EXP_NAME=${1:-"train_DTU_2L_16H"}
SAMPLE=${2:-"scan23"}

python3 generator.py --config configs/render_experiment.txt --expname render_experiment_${SAMPLE} --generate_specific_samples ${SAMPLE} --gen_pose 0 --no_parallel --ckpt_expname ${EXP_NAME}
