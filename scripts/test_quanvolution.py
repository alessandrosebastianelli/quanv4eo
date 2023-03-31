import sys
sys.path += ['./', '../']


from layers.QConv2D import QConv2D

import numpy as np
import argparse
import time

import jax; jax.config.update('jax_platform_name', 'cpu')

import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

def quanvolution(IMG_SHAPE, QUBITS, FILTERS, KERNEL_SIZE, STRIDE, NUM_JOBS, CIRCUIT):

    conv1 = QConv2D(
        QUBITS,
        FILTERS, 
        KERNEL_SIZE, 
        STRIDE, 
        parallelize = NUM_JOBS,
        circuit = CIRCUIT
    )

    img = np.zeros(IMG_SHAPE)
    
    for i in range(10):
        if i == 1:
            start = time.time()
        out = conv1.apply(img)
    end = time.time()
    
    t = (end-start)/9
    return t

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply Quanvolution')
    
    parser.add_argument('--img_shape', nargs='+', type=int, required=True)
    parser.add_argument('--qubits', type=int, required=True)
    parser.add_argument('--filters', type=int, required=True)
    parser.add_argument('--kernel_size', type=int, required=True)
    parser.add_argument('--stride', type=int, required=True)
    parser.add_argument('--num_jobs', type=int, required=True)
    parser.add_argument('--circuit', required=True)
    
    args = parser.parse_args()
    
    t = quanvolution(tuple(args.img_shape), args.qubits, args.filters, args.kernel_size, args.stride, args.num_jobs, args.circuit)
    
    f = open('results.csv', 'a')
    f.write('{},{},{},{},{},{},{},{}\n'.format(str(tuple(args.img_shape)), args.qubits, args.filters, args.kernel_size, args.stride, args.num_jobs, args.circuit, t))