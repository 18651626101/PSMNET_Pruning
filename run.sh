#!/bin/bash

python main.py --maxdisp 192 \
               --model stackhourglass \
               --datapath /home/jump/dataset \
               --epochs 10 \
               --loadmodel ./trained/checkpoint_10.tar \
               --savemodel ./trained/



python finetune.py --maxdisp 192 \
                   --model stackhourglass \
                   --datatype 2015 \
                   --datapath /home/jump/dataset/kitti2015/training/ \
                   --epochs 300 \
                   --loadmodel ./pretrained/pretrained_sceneflow.tar \
                   --savemodel ./trained/

