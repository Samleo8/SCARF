#!/bin/bash

EXP_NAME=${1:-"train_DTU_2L_32H"}
SCAN_NUM=${2:-23}

echo "Viewing 3D reconstruction of scan $SCAN_NUM for $EXP_NAME in Python"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICE python3 view_3d_reconstruction.py ${EXP_NAME} ${SCAN_NUM}