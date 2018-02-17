# Graph signals layers for Tensorflow

This project is to implement several graph signals algorithms for use in a deep learning framework. The plan is to do this for point clouds initially. 

## Use 

This is meant to be able to be used in any layer of the network. If you want to preproccess the data with some convolutional layers and maybe a dense layer, then implement a semisupervised graph algorithm, this should accomplish that. This requires an op class that stores the data in a linear array with an `add` op and then, once we have enough we compile this into an M/KD/cover-tree. Then we can make fast queries and use graph convolutional algorithms at speed, after the network has extracted some features.

## Issues

The current implementation uses a linear search over an array. This is not ideal. This needs to be replaced with an M-tree, internal to tensorflow and a set of tightly. This will require an extension written in C++. 
