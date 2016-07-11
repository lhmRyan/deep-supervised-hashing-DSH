#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train \
    --solver=NUS/finetune_solver.prototxt \
    --weights=NUS/model_4.caffemodel #>>NUS/log_finetune.txt 2>&1
