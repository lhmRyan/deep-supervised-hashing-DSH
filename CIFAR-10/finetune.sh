#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train \
    --solver=CIFAR-10/finetune_solver.prototxt \
    --weights=CIFAR-10/model_3_1.caffemodel #\
    #>>CIFAR-10/log.txt 2>&1 
