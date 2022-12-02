#!/bin/bash

wget http://roboimagedata2.compute.dtu.dk/data/MVS/Rectified.zip -P data/
unzip data/Rectified.zip -d data/
mv data/Rectified/* data/DTU_MVS
rmdir data/Rectified
