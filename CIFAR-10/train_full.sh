#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train \
    --solver=CIFAR-10/solver.prototxt #--weights=DNNH/cifar10_iter_50000.caffemodel #>>CIFAR-10/log.txt 2>&1 #\
#    --snapshot=CIFAR-10/cifar10_iter_30000.solverstate
