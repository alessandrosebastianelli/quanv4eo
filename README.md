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
2. [Multi Quantum Convolutional Processing](MultiQuantumConvolutionalProcessing.ipynb): this notebook shows how to load a dataset, how to apply quantum convolution with more quantum kernels to precces it and how to save results
3. [Quantum Convolutional Neural Networks](QuantumConvolutionalNeuralNetworks.ipynb): this notebook loads the quantum processed dataset and run the classification of the dataset


Similary there are python file that coresponds to the notebooks:

1. [Quantum Convolutional Processing](qconv-1cXdataset.py): this python script shows how to load a dataset, how to apply quantum convolution to precces it and how to save results
2. [Multi Quantum Convolutional Processing](qconv-1cXclass.py): this python script shows how to load a dataset, how to apply quantum convolution with more quantum kernels to precces it and how to save results

### Citation

```
@ARTICLE{sebastianelli2025quanv4eo,
  author={Sebastianelli, Alessandro and Mauro, Francesco and Ciabatti, Giulia and Spiller, Dario and Le Saux, Bertrand and Gamba, Paolo and Liberata Ullo, Silvia},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Quanv4EO: Empowering Earth Observation by Means of Quanvolutional Neural Networks}, 
  year={2025},
  volume={63},
  number={},
  pages={1-15},
  keywords={Feature extraction;Quantum computing;Quantum circuit;Filters;Qubit;Integrated circuit modeling;Remote sensing;Data mining;Climate change;Deep learning;Convolutional neural networks;Earth Observing System;Earth observation (EO);quantum computing (QC);quantum convolutional neural networks (CNNs);quantum deep learning (DL);remote sensing (RS)},
  doi={10.1109/TGRS.2025.3556335}
}
```

### References

[1] Henderson, M., Shakya, S., Pradhan, S., & Cook, T. (2020). Quanvolutional neural networks: powering image recognition with quantum circuits. Quantum Machine Intelligence, 2(1), 1-9.

