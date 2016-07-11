#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train \
    --solver=NUS/solver.prototxt -gpu 1,2 --snapshot=NUS/nus_iter_140000.solverstate #>>NUS/log_12bit.txt 2>&1 #\
#    --weights=NUS/cifar10_iter_20000.caffemodel
#    --snapshot=NUS/nus_iter_70000.solverstate


#$TOOLS/caffe train \
#    --solver=CIFAR-10/cifar10_full_solver_lr1.prototxt \
#    --snapshot=CIFAR-10/cifar10_full_iter_40000.solverstate

#$TOOLS/caffe train \
#    --solver=CIFAR-10/cifar10_full_solver_lr2.prototxt \
#    --snapshot=CIFAR-10/cifar10_full_iter_160000.solverstate
