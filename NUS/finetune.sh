#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train \
    --solver=NUS/finetune_solver.prototxt \
    --weights=NUS/nus_iter_150000.caffemodel #>>NUS/log_finetune.txt 2>&1
