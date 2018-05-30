#!/bin/bash -v

source /home/luban/.bashrc
source /etc/profile

source /home/luban/miniconda3/bin/activate base

python apollo.py trial --model=coco_backbone

