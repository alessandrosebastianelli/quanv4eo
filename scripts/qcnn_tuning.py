import sys
sys.path += ['./', '../']
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np
import keras_tuner
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

# Loading the dataset
dataset_name = 'EuroSAT_processed_v2_QCNN_0'
root = os.path.join('/Users/asebastianelli/Desktop/quanvolutional4eo/datasets', dataset_name)
dhandler = datahandler(root)
train_set, val_set = dhandler.split(None, factor=0.2)
labels_mapper, x_t, y_t = dhandler.unpack(train_set)
labels_mapper, x_v, y_v = dhandler.unpack(val_set)
shape = np.load(x_t[0]).shape
classes = dhandler.paths.keys()

# Training and Validation data loader
x_train, y_train, = datareader.generatorv3([x_t, y_t], shape, normalize=False)
x_val, y_val   = datareader.generatorv3([x_v, y_v], shape, normalize=False)

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)

# Early Stopping to avoid overfitting
es = EarlyStopping(monitor='val_loss', patience=5, mode='auto', verbose=0, baseline=None)

def build_model(hp):
    learning_rate  = hp.Float('learning_rate', min_value = 0.0002, max_value = 0.2, sampling='log')
    #batch_size     = hp.Choice('batch_size', [64,16])

    dense1         = hp.Int('dense1', min_value=64, max_value=128, step = -32) #[[32,16], [64,32,16], [128,64,32,16]])
    dense2         = hp.Int('dense2', min_value=32, max_value=64, step = -32) #[[32,16], [64,32,16], [128,64,32,16]]) 
    dense3         = hp.Int('dense3', min_value=16, max_value=64, step = -8) #[[32,16], [64,32,16], [128,64,32,16]])
    
    conv1          = hp.Int('conv1', min_value=16, max_value=32, step = 16) #[[32,16], [64,32,16], [128,64,32,16]])
    conv2          = hp.Int('conv2', min_value=32, max_value=64, step = 16) #[[32,16], [64,32,16], [128,64,32,16]]) 
    conv3          = hp.Int('conv3', min_value=64, max_value=128, step = 16) #[[32,16], [64,32,16], [128,64,32,16]])

    dropout        = hp.Float('dropout', min_value = 0.1, max_value = 0.5, sampling='log')
    kernel         = hp.Choice('kernel',[3,4,5])
    stride         = hp.Choice('stride',[1,2])
    pool_size      = hp.Choice('pool_size',[1,2])
    pool_stride    = hp.Choice('pool_stride',[1,2])
    padding        = hp.Choice('padding',['valid','same'])

    qcnn = QCNNv1(img_shape = shape, n_classes = 10)
                
    # Forcing current hyperparameters
    qcnn.learning_rate  = learning_rate
    #qcnn.batch_size     = batch_size
    qcnn.dense          = [dense1, dense2, dense3]
    qcnn.conv           = [conv1, conv2, conv3]
    qcnn.epochs         = 200
    qcnn.early_stopping = 5
    qcnn.dropout        = dropout
    qcnn.kernel         = kernel
    qcnn.stride         = stride
    qcnn.pool_size      = pool_size
    qcnn.pool_stride    = pool_stride
    qcnn.padding        = padding

    qcnn.model = qcnn.build()

    return qcnn.model

tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective="val_accuracy",
    max_trials=10,
    executions_per_trial=2,
    overwrite=False,
    directory=os.path.join('results', 'dltuning'),
    project_name="QCNN-Tuning",
)

print(tuner.search_space_summary())

tuner.search(x_train, y_train, epochs=200, validation_data=(x_val, y_val), callbacks=[es], batch_size=64, shuffle=True)