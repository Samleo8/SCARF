#!/bin/bash

EXP_NAME=${1:-"train_DTU"}

# Move renderings images to special folder
# NOTE: folders are of form epoch_1000_scan23_0/rendering.png
BASE_RENDER_FOLDER=./logs/${EXP_NAME}/renderings
mkdir -p $BASE_RENDER_FOLDER

BASE_VIS_FOLDER=./logs/${EXP_NAME}/training_visualization
FOLDERS=$(ls ${BASE_VIS_FOLDER})
for FOLDER in $FOLDERS; do
    if [ ! -d ${BASE_VIS_FOLDER}/${FOLDER} ]; then
        continue
    fi

    # Get epoch number
    EPOCH=$(echo $FOLDER | cut -d'_' -f2)

    # Get it as a sequence for video processing
    STEP_SIZE=1000
    SEQ=$((EPOCH / $STEP_SIZE))
    SEQ=$(printf "%05d\n" $SEQ)

    # Get scan number
    SCAN=$(echo $FOLDER | cut -d'_' -f3)
    # Get view number
    POSE=$(echo $FOLDER | cut -d'_' -f4)

    echo "Copying rendering for epoch $EPOCH, scan $SCAN, pose $POSE, as sequence $SEQ"

    # Create folder
    RENDER_OUT_FOLDER=${BASE_RENDER_FOLDER}/${SCAN}_${POSE}/
    mkdir -p $RENDER_OUT_FOLDER

    # Copy rendering
    cp ${BASE_VIS_FOLDER}/${FOLDER}/rendering.png ${RENDER_OUT_FOLDER}/${SEQ}.png
done
