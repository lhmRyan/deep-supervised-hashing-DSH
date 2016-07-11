#This is the README information of the following publication
========================================================================
Deep Supervised Hashing for Fast Image Retrieval,

Version 1.0,  Copyright(c) May, 2016

Haomiao Liu, Ruiping Wang, Shiguang Shan, Xilin Chen.

All Rights Reserved.

-------------------------------------------------------------------------
 
Example usage:

1. Modify "Makefile.config" according to your system, and follow the 
   instructions on "http://caffe.berkeleyvision.org/installation.html"
   to compile the source code.

2. Run "data/cifar10/get_cifar10.sh" to download the data of CIFAR-10.

3. Run "CIFAR-10/create_cifar10.sh" to convert the data of CIFAR-10
   to LMDB format.

4. Run "CIFAR-10/train_full.sh" to train an example model.

5. Run "CIFAR-10/finetune.sh" to finetune an existing model 
   for different code length.

-------------------------------------------------------------------------

Tips:

1. We have provided a tool for extracting network activations, you may
   refer to "CIFAR-10/extract_code.sh" for example usage. The example 
   code extracts the real-valued network outputs, and writes them into a 
   binary file. The first 12 floating point numbers correspond to the
   first image, and the second 12 correspond to the second image, and 
   so on.

2. We have also provided a "HashingImageData" layer which corresponds to 
   the online image pair generation scheme described in our paper. This 
   scheme is extremely useful for large scale dataset, where dissimilar
   pairs outnumber similar pairs. Please refer to 
   "CIFAR-10/train_test.prototxt" for example usage.

3. Our code was modified to support multi-label images as in NUS-WIDE, 
   please refer to "NUS/train.txt", "NUS/test.txt", "NUS/create_NUS.sh"
   for example usages.

-------------------------------------------------------------------------
Please refer to the following paper if you find the source code helpful:

Haomiao Liu, Ruiping Wang, Shiguang Shan, Xilin Chen.

Deep Supervised Hashing for Fast Image Retrieval

In Proc. CVPR 2016.

Contact: haomiao.liu@vipl.ict.ac.cn

========================================================================