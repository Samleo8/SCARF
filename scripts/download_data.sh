#!/bin/bash

# clear old zip files before unzip
rm data/Rectified.zip*

# download data from internet
wget http://roboimagedata2.compute.dtu.dk/data/MVS/Rectified.zip -P data/
unzip data/Rectified.zip -d data/
mv data/Rectified/* data/DTU_MVS
rmdir data/Rectified

# clear old zip files after unzip
rm data/Rectified.zip*
