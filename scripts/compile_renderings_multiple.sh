#!/bin/bash

EXPERIMENTS=(train_DTU train_DTU_2L_16H train_DTU_2L_nocompress)

for EXP in ${EXPERIMENTS[@]}; do
    echo "Compiling renderings for $EXP"
    ./scripts/compile_renderings.sh $EXP &
done