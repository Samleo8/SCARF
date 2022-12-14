#!/bin/bash

LOG_DIR=./logs
DOCS_DIR=./docs/videos

for DIR in $LOG_DIR/train_DTU_*
do
    if [ ! -d $DIR ]; then
        continue
    fi

    BASENAME=$(basename $DIR)
    ARCHI_NAME=${BASENAME#train_DTU_}

    cp $DIR/renderings/scan23_0/render.mp4 $DOCS_DIR/render23_$ARCHI_NAME.mp4
done