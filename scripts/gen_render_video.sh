#!/bin/bash

EXP_NAME=$1
FPS=${2:-10}

if [ -z $EXP_NAME ]; then
    echo "USAGE: ./mp4-from-folder.sh <subfolder-output-name> [results-folder]"
    exit
fi

RENDER_BASE_FOLDER=./logs/$EXP_NAME/renderings
for SUBFOLDER in $RENDER_BASE_FOLDER/*; do
    echo "Processing $SUBFOLDER"
    RENDER_FOLDER=$SUBFOLDER
    ffmpeg -y -framerate $FPS -i $RENDER_FOLDER/%0005d.png -loop -1 -profile:v high -crf 28 -pix_fmt yuv420p $RENDER_FOLDER/render.mp4
    cd -
done
