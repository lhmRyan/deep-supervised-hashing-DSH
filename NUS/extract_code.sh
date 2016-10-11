#!/usr/bin/env sh

rm NUS/code.dat
rm NUS/label.dat

build/tools/extract_features_binary NUS/nus_iter_150000.caffemodel NUS/train_test.prototxt ip1 NUS/code.dat 100
build/tools/extract_features_binary NUS/nus_iter_150000.caffemodel NUS/train_test.prototxt label NUS/label.dat 100