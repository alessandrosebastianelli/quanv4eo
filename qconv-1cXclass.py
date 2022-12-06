from data.datahandler import datahandler
from data.datareader import datareader
from layers.QConv3D import QConv3D
from layers.QConv2D import QConv2D
from datetime import datetime
from circuits.random import *
from utils.plotter import *

from tqdm.auto import tqdm
import numpy as np
import os



datasets_path = ['euro1', 'euro2', 'euro3', 'euro4', 'euro5', 'euro6', 'euro7', 'euro8', 'euro9', 'euro10']


for i, dataset_name in enumerate(datasets_path):
    print('{:.^100}\n'.format(' Dataset '+str(dataset_name)+' '))

    #==========================================================================================
    # Load the dataset 
    #==========================================================================================
    root = os.path.join('datasets', dataset_name)
    dhandler = datahandler(root)
    dhandler.print_report(name=dataset_name)
    labels_mapper, x, y = dhandler.unpack(dhandler.paths)
    classes = dhandler.paths.keys()
    loader  = datareader.generatorv2((x, y), (64,64,3))

    #==========================================================================================
    # Define the quantum circuit
    #==========================================================================================
    QUBITS      = 9
    KERNEL_SIZE = 3
    FILTERS     = 8
    N_LAYERS    = 1
    STRIDE      = 2
    NUM_JOBS    = 16

    SEED = i
    np.random.seed(SEED)

    circuits = [ry_random(QUBITS, KERNEL_SIZE, FILTERS, N_LAYERS, SEED)]

    conv1 = QConv2D(
        circuits,
        FILTERS, 
        KERNEL_SIZE, 
        STRIDE, 
        NUM_JOBS
    )
    
    #==========================================================================================
    # Apply the Quantum circuit
    #==========================================================================================
    new_name = dataset_name+'_processed_'+datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
    # Create folder structures
    root2 = root.replace(dataset_name, new_name)
    for path in dhandler.paths.keys():
        os.makedirs(os.path.join(root2, path), exist_ok=True)

    # Apply Quantum Convolution to the dataset
    gen = iter(datareader.generatorv2((x, y), (64,64,3)))
    for i in tqdm(range(len(x)), colour='black'):
        (xi, yi, pi) = next(gen)
        out1 = conv1.apply(xi, verbose=True)
        pi = pi.replace(dataset_name, new_name)
        pi = pi.replace('.'+pi.split('.')[-1], '.npy')
        # Save the results
        with open(pi, 'wb') as f:
            np.save(f, out1)