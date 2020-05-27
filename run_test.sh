#!/bin/bash

python submission.py --KITTI 2015 \
               --datapath /home/jump/dataset/kitti2015/testing/ \
               --loadmodel ./pretrained/pretrained_model_KITTI2015.tar \
               --model stackhourglass 
