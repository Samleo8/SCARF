#!/bin/bash

EXP_NAMES=(train_DTU_2L_32H train_DTU_4L_16H train_DTU_2L_16H_reduced)

CNT=0
PIDS=""
for EXP_NAME in "${EXP_NAMES[@]}"; do
    echo "Processing $EXP_NAME on GPU $CNT"

    ./scripts/render_finetune.sh ${EXP_NAME} "scan23" $CNT &
    PIDS="$! $PIDS"

    CNT=$((CNT + 1))
done

wait $PIDS
echo "Done!"
