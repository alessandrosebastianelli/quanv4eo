from pennylane.templates import RandomLayers
from pennylane import numpy as np
import pennylane as qml
from torch import nn
import torchvision
import torch

class QonvLayer(nn.Module):
    '''
        Quantum Convolution 2D
    '''

    def __init__(self, kernel_size =2, stride=2, device="default.qubit", wires=4, circuit_layers=4, n_rotations=8, out_channels=4, seed=None):
        super(QonvLayer, self).__init__()
        # init device
        self.wires = wires
        self.dev = qml.device(device, wires=self.wires)
        self.stride = stride
        self.kernel_size = kernel_size
        self.out_channels = min(out_channels, wires)
        
        if seed is None:
            seed = np.random.randint(low=0, high=10e6)
        
        # random circuits
        @qml.qnode(device=self.dev)
        def circuit(inputs, weights):            
            # Encoding of 4 classical input values
            for j in range(self.kernel_size**2):
                qml.RY(inputs[j], wires=j)
            # Random quantum circuit
            RandomLayers(weights, wires=list(range(self.wires)), seed=seed)
            
            # Measurement producing 4 classical output values
            return [qml.expval(qml.PauliZ(j)) for j in range(self.out_channels)]
        
        weight_shapes = {"weights": [circuit_layers, n_rotations]}
        self.circuit = qml.qnn.TorchLayer(circuit, weight_shapes=weight_shapes)

    def forward(self, img):

        '''
            Quantum convolution.

            Inputs:
                - image: a 3D channel-last matrix in R^(w,h,c), w: width, h: height, c:channels
            Output:
                - q_results: 1D quantum output vector
        '''
        # Calculates the image shape after the convolution
        bs, h, w, ch = img.shape
        if ch > 1:
            img = img.mean(axis=-1).reshape(bs, h, w, 1)

        h_out = (h-self.kernel_size) // self.stride +1
        w_out = (w-self.kernel_size) // self.stride +1

        out = torch.zeros((bs, h_out, w_out, self.out_channels))

        for b in range(bs):
            # Spatial Loops                                                
            for j in range(0, h_out, self.stride):
                for k in range(0, w_out, self.stride):            
                    # Process a kernel_size*kernel_size region of the images
                    # with the quantum circuit stride*stride
                    p = img[b, j:j+self.kernel_size, k:k+self.kernel_size, 0]

                    q_results = self.circuit(torch.Tensor(p.reshape(-1)))

                    for c in range(self.out_channels):
                        out[b, j // self.kernel_size, k // self.kernel_size, c] = q_results[c]

        return out