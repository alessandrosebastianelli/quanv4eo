import sys
sys.path += ['./', '../']

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import logging
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# Suppress warnings
import warnings
def fxn(): warnings.warn("deprecated", DeprecationWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

from utils.converter import convert_labels_mapper
from data.datahandler import datahandler
from data.datareader import datareader
from utils.plotter import *
from models.QCNN import *

#------------------------------------------------------------------------------------------------------------
#                                          HYPERPARAMETERS TUNING
#-------------------------------------------------------------------------------------------------------------
# Hyperparameters to be tuned, the following is the base parameter dictionay.
#
# cnnv1s = {
#            'loss':            'categorical_crossentropy',
#            'learning_rate':   0.0002,
#            'metrics':         ['accuracy'],
#            'dropout':         0.2,
#            'batch_size':      16,
#            'epochs':          100,
#            'early_stopping':  30,
#            'dense':           [256, 128, 64, 32, 16],    # Vector of #neurons for each dense layer
#            'conv':            [12, 64, 128, 256],        # Vector of #filters for each convolution layer
#            'kernel':          3,                         # Kernel Size for the convolution
#            'stride':          1,                         # Strides for the convolution
#            'pool_size':       2,                         # Kernel Size for the Global Average Pooling
#            'pool_stride':     1,                         # Strides for the Global Average Pooling
#        }

# In this case we are tuning the model using the following hyperparameters lists
#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! WARNING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
# Be aware that exploring 3 different hyperparameters takes a large amount of time.
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

learning_rate   = [0.002]
batch_size      = [64]
dense           = [[32,16], [64,32,16], [128,64,32,16]]
conv            = [[32,64],[32,64,128]]
dropout         = [0.1,0.2, 0.5, 0.8]
kernel          = [3]
stride          = [1]
pool_size       = [2]
pool_stride     = [2]

# Loading the dataset
dataset_name = 'EuroSAT_processed_v2_QCNN_0'
root = os.path.join('/Users/asebastianelli/Desktop/quanvolutional4eo/datasets', dataset_name)
dhandler = datahandler(root)
train_set, val_set = dhandler.split(None, factor=0.2)
labels_mapper, x_t, y_t = dhandler.unpack(train_set)
labels_mapper, x_v, y_v = dhandler.unpack(val_set)
shape = np.load(x_t[0]).shape
classes = dhandler.paths.keys()

VERBOSE = 0
# Start the hyperparameters tuning
for lr in tqdm(learning_rate, desc = '{:<30s}'.format('Learning Rate')):
    for bs in tqdm(batch_size, leave=False, desc = '{:<30s}'.format('Batch Size')):
        for d in tqdm(dense, leave=False,   desc = '{:<30s}'.format('Dense Layers')):
            for c in tqdm(conv, leave=False, desc = '{:<30s}'.format('Conv Layers')):
                for do in tqdm(dropout, leave=False, desc = '{:<30s}'.format('Dropout')):
                    for k in tqdm(kernel, leave=False, desc = '{:<30s}'.format('Kernel Size')):
                        for s in tqdm(stride, leave=False, desc = '{:<30s}'.format('Stride')):
                            for pk in tqdm(pool_size, leave=False, desc = '{:<30s}'.format('Pool Size')):
                                for ps in tqdm(pool_stride, leave=False, desc = '{:<30s}'.format('Pool Stide')):
                                    qcnn = QCNNv1(img_shape = shape, n_classes = 10)
                
                                    # Forcing current hyperparameters
                                    qcnn.learning_rate  = lr
                                    qcnn.batch_size     = bs
                                    qcnn.dense          = d
                                    qcnn.conv           = c
                                    qcnn.epochs         = 200
                                    qcnn.early_stopping = 5
                                    qcnn.dropout        = do
                                    qcnn.kernel         = k
                                    qcnn.stride         = s
                                    qcnn.pool_size      = pk
                                    qcnn.pool_stride    = ps

                                    qcnn.model = qcnn.build()

                                    if VERBOSE: print(qcnn.model.summary())
                                    
                                    # Train and test the model
                                    qcnn.train_test([x_t, y_t], [x_v, y_v], convert_labels_mapper(labels_mapper), normalize = None, verbose = VERBOSE)
                                    
                                    # Plot training curvers
                                    plot_training('QCNNv1', display = False, verbose=VERBOSE)