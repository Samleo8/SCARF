#!/bin/bash

scpfrom() {
    awsscpfrom $@
    # gscpfrom $@
}

dl_log() {
    LOG_TYPE=${1:-"all"}
    EXP_NAME=${2:-"train_DTU"}
    INSTANCE=${3:-"main"}

    case $LOG_TYPE in
    "args" | "config" | "metadata")
        scpfrom ~/vl/project/stereo-nerf/logs/${EXP_NAME}/args.txt ./logs/${EXP_NAME}/ $INSTANCE
        scpfrom ~/vl/project/stereo-nerf/logs/${EXP_NAME}/config.txt ./logs/${EXP_NAME}/ $INSTANCE
        ;;
    "tensorboard")
        scpfrom ~/vl/project/stereo-nerf/logs/${EXP_NAME}/tensorboard ./logs/${EXP_NAME}/ $INSTANCE
        ;;
    "training_visualization" | "vis")
        gscpfrom ~/vl/project/stereo-nerf/logs/${EXP_NAME}/training_visualization ./logs/${EXP_NAME}/ $INSTANCE
        ;;
    "render" | "renderings")
        # Download renderings
        # NOTE: Need to run script ./scripts/compile_renderings from within system to do so properly
        scpfrom ~/vl/project/stereo-nerf/logs/${EXP_NAME}/renderings ./logs/${EXP_NAME} $INSTANCE
        ;;
    "all")
        dl_log metadata
        dl_log tensorboard
        dl_log training_visualization
        ;;
    *.tar)
        # Download a specific checkpoint
        CKPT_ID=${LOG_TYPE%.tar}
        CKPT_ID=$(expr $CKPT_ID + 0)
        echo "Downloading checkpoint $CKPT_ID"
        scpfrom ~/vl/project/stereo-nerf/logs/${EXP_NAME}/$LOG_TYPE ./logs/${EXP_NAME}/ $INSTANCE

        # Download the corresponding visualizations
        scpfrom ~/vl/project/stereo-nerf/logs/${EXP_NAME}/training_visualization/epoch_${CKPT_ID}_scan* ./logs/${EXP_NAME}/training_visualization $INSTANCE
        ;;
    *)
        dl_log "all"
        ;;
    esac
}

# Script to download logs from the server
LOG_TYPE=${1:-"all"}
EXP_NAME=${2:-"train_DTU_4L_16H"}
INSTANCE=${3:-"main"}

mkdir -p logs/${EXP_NAME}
mkdir -p logs/${EXP_NAME}/training_visualization
mkdir -p logs/${EXP_NAME}/tensorboard

dl_log $LOG_TYPE $EXP_NAME $INSTANCE
