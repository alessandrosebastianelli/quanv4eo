# Quantum Convolutional Neural Networks

This repository contains a novel implementation of Quanvolutional Neural Network presented by [1]. 

With respect to [1], this implementation allows to:

-  change the quantum circuit strcuture
-  change the kernel size, the stride and the number of output filters
-  use image with more channels
-  stack multiple Quanvolutional layers
-  to use more quantum circuits (to make a parallel it means to apply more quantum kernels to one image)

This implementation contains both Conv2D and Conv2D quantum layers, both can be used in normal mode and in parallel (to speed up the processing).


## Contents

The repository contains several jupyter notebook that must be runned in a precise order

1. [Quantum Convolutional Processing](QuantumConvolutionalProcessing.ipynb): this notebook shows how to load a dataset, how to apply quantum convolution to precces it and how to save results
2. [Quantum Convolutional Neural Networks](QuantumConvolutionalNeuralNetworks.ipynb): this notebook loads the quantum processed dataset and run the classification of the dataset
3.  


k### *References*

[1] Henderson, M., Shakya, S., Pradhan, S., & Cook, T. (2020). Quanvolutional neural networks: powering image recognition with quantum circuits. Quantum Machine Intelligence, 2(1), 1-9.

