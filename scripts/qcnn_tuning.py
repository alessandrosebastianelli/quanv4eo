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
learning_rate   = [0.0002]
batch_size      = [16]
dense           = [[32,16], [64,32,16], [128,64,32,16]]
conv            = [[32,64],[32,64,128]]

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
for lr in tqdm(learning_rate):
    for bs in tqdm(batch_size, leave=False):
        for d in tqdm(dense, leave=False):
            for c in tqdm(conv, leave=False):
                qcnn = QCNNv1(img_shape = shape, n_classes = 10)
                
                # Forcing current hyperparameters
                qcnn.learning_rate  = lr
                qcnn.batch_size     = bs
                qcnn.dense          = d
                qcnn.conv           = c
                qcnn.epochs         = 200
                qcnn.early_stopping = 10

                qcnn.model = qcnn.build()

                if VERBOSE: print(qcnn.model.summary())
                
                # Train and test the model
                qcnn.train_test([x_t, y_t], [x_v, y_v], convert_labels_mapper(labels_mapper), normalize = None, verbose = VERBOSE)
                
                # Plot training curvers
                plot_training('QCNNv1', display = False, verbose=VERBOSE)