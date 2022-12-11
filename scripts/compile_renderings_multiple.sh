#!/bin/bash

EXPERIMENTS=(train_DTU_2L_32H train_DTU_4L_16H train_DTU_2L_16H_reduced)

for EXP in ${EXPERIMENTS[@]}; do
    echo "Compiling renderings for $EXP"
    ./scripts/compile_renderings.sh $EXP &
done
