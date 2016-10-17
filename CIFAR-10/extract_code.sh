#!/usr/bin/env sh

rm CIFAR-10/code.dat
rm CIFAR-10/label.dat

build/tools/extract_features_binary CIFAR-10/cifar10_iter_70000.caffemodel CIFAR-10/train_test.prototxt ip1 CIFAR-10/code.dat 100 0
build/tools/extract_features_binary CIFAR-10/cifar10_iter_70000.caffemodel CIFAR-10/train_test.prototxt label CIFAR-10/label.dat 100 0

