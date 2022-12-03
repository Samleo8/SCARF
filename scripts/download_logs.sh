#!/bin/bash

INSTANCE=${1:-power}

mkdir -p logs/train_DTU
mkdir logs/train_DTU/training_visualization

gscpfrom ~/vl/project/stereo-nerf/logs/train_DTU/args.txt ./logs/train_DTU/ $INSTANCE
gscpfrom ~/vl/project/stereo-nerf/logs/train_DTU/config.txt ./logs/train_DTU/ $INSTANCE
gscpfrom ~/vl/project/stereo-nerf/logs/train_DTU/training_visualization ./logs/train_DTU/training_visualization $INSTANCE
gscpfrom ~/vl/project/stereo-nerf/logs/train_DTU/tensorboard ./logs/train_DTU/tensorboard $INSTANCE