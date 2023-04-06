import sys
sys.path += ['./', '../']

from data.datahandler import datahandler
from data.datareader import datareader
from layers.QConv2D import QConv2D
from utils import test_loader
from utils.plotter import *

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import pennylane as qml
import numpy as np
import os

import jax; jax.config.update('jax_platform_name', 'gpu')

# Suppress warnings
import warnings
def fxn(): warnings.warn("deprecated", DeprecationWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

#=====================================================================    
#-------------------------- USER PARAMETERS --------------------------
#=====================================================================
dataset_name = 'EuroSAT'
QUBITS      = 16
KERNEL_SIZE = 4
FILTERS     = 12
N_LAYERS    = 2
STRIDE      = 4
NUM_JOBS    = 16
SEED        = 1
MODE        = 'QCNN' # 'CNN' or 'QCNN'
CIRCUIT     = 'rxyz' # 'rxyz' or 'rx' or 'ry' or 'rz'
NCONV       = 2
SHAPE       = (64,64,3)
new_name = dataset_name+'_processed_'+MODE

os1 = SHAPE[0]
print('Initial Shape == ', os1)
for n in range(NCONV):
    os1 = (os1 - KERNEL_SIZE)/STRIDE+1
    print('Shape at layer {} == {}'.format(n+1, os1))
print()


#=====================================================================    
#---------------------------- LOAD DATASET ---------------------------
#=====================================================================
root = os.path.join('../datasets', dataset_name)
dhandler = datahandler(root)
dhandler.print_report(name=dataset_name)
labels_mapper, x, y = dhandler.unpack(dhandler.paths)

print('Labels')
for key in labels_mapper: print('{:<30s}{}'.format(key,labels_mapper[key]))

print('\nDataset Size')
print('{:<30s}{}'.format('Images', len(x)))

print('\nTraining Dataset samples')
print('{:<30s}{}'.format('X Train', x[0]))
print('{:<30s}{}'.format('X Train', y[0]))
classes = dhandler.paths.keys()

#=====================================================================    
#----------------------------- PROCESSING ----------------------------
#=====================================================================
if MODE == 'CNN':
    cnn = Sequential()
    l = Conv2D(
        filters     = FILTERS, 
        kernel_size = KERNEL_SIZE,
        strides     = STRIDE,
        activation  = None,
        input_shape = SHAPE
    )
    l.trainable = False
    cnn.add(l)
    cnn.compile(loss='mse', optimizer='adam')

if MODE == 'QCNN':
    conv1 = QConv2D(
            QUBITS,
            FILTERS, 
            KERNEL_SIZE, 
            STRIDE, 
            parallelize = NUM_JOBS,
            circuit = CIRCUIT
        )

# Create folder structures
for n in range(NCONV):
    root2 = root.replace(dataset_name, new_name+'_{}'.format(n))
    for path in dhandler.paths.keys():
        os.makedirs(os.path.join(root2, path), exist_ok=True)

# Apply Quantum or Classical Convolution to the dataset
gen = iter(datareader.generatorv2((x, y), SHAPE))
for i in tqdm(range(len(x)), colour='black'):
    (xi, yi, pt) = next(gen)
    for n in range(NCONV):
        if MODE=='CNN': xi = cnn.predict(xi[None,...], verbose = False)[0,...]
        if MODE=='QCNN': xi = conv1.apply(xi, verbose = False)
        # Create Image path
        pi = pt
        pi = pi.replace(dataset_name, new_name+'_{}'.format(n))
        pi = pi.replace('.'+pi.split('.')[-1], '.npy')
        # Save the results
        with open(pi, 'wb') as f:
            np.save(f, xi)