#!/bin/bash

dl_log() {
    LOG_TYPE=${1:-"all"}
    INSTANCE=${2:-"power"}

    case $LOG_TYPE in
    "args" | "config" | "metadata")
        gscpfrom ~/vl/project/stereo-nerf/logs/train_DTU/args.txt ./logs/train_DTU/ $INSTANCE
        gscpfrom ~/vl/project/stereo-nerf/logs/train_DTU/config.txt ./logs/train_DTU/ $INSTANCE
        ;;
    "tensorboard")
        gscpfrom ~/vl/project/stereo-nerf/logs/train_DTU/tensorboard ./logs/train_DTU/ $INSTANCE
        ;;
    "training_visualization" | "vis")
        gscpfrom ~/vl/project/stereo-nerf/logs/train_DTU/training_visualization ./logs/train_DTU/ $INSTANCE
        ;;
    "all")
        dl_log metadata
        dl_log tensorboard
        dl_log training_visualization
        ;;
    *.tar)
        # Download a specific checkpoint
        echo "Downloading checkpoint $LOG_TYPE"
        gscpfrom ~/vl/project/stereo-nerf/logs/train_DTU/$LOG_TYPE ./logs/train_DTU/ $INSTANCE
        ;;
    *)
        dl_log "all"
        ;;
    esac
}

LOG_TYPE=${1:-all}
INSTANCE=${2:-power}

mkdir -p logs/train_DTU
mkdir -p logs/train_DTU/training_visualization
mkdir -p logs/train_DTU/tensorboard

dl_log $LOG_TYPE $INSTANCE
