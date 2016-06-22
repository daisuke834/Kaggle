#!/bin/bash
echo Start
cd ~/Kaggle/FacialKeypointsDetection
pwd
python ./keras_cnn.py >& ./output/`date "+%Y_%m_%d_%H%M%S"`_keras_cnn.txt
echo Program Ends
echo Sync outputs
aws s3 sync ./output s3://kaggle928374/FacialKeypointsDetection/output/
#sudo shutdown -h now

