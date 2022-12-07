#!/bin/bash

EXPERIMENTS=(train_DTU train_DTU_2L_16H train_DTU_2L_nocompress)

for EXP in ${EXPERIMENTS[@]}; do
    ./scripts/download_logs.sh render $EXP
    ./scripts/gen_render_video.sh $EXP
done

