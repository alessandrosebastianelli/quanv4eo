import sys
sys.path += ['./', '../']


from layers.QConv2D import QConv2D

import numpy as np
import argparse
import time
import os

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
    
    start = time.time()
    for i in range(100):
        out = conv1.apply(img)
    end = time.time()
    
    t = (end-start)/100
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
    
    res_path = os.path.join('results, timing_results.csv')
    if not os.path.isfile(res_path):
        colnames = 'imgsize,qubits,filters,kernelsize,stride,numjobs,circuit,time\n'
        f = open(res_path, 'a')
        f.write(colnames)
        f.close()
            
    f = open(res_path, 'a')
    f.write('{},{},{},{},{},{},{},{}\n'.format(args.img_shape[0], args.qubits, args.filters, args.kernel_size, args.stride, args.num_jobs, args.circuit, t))
    f.close()