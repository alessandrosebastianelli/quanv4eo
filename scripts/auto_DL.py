from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import autokeras as ak
import pandas as pd
import numpy as np
import glob
import os

import sys
sys.path += ['./', '../']


#=================================================================
# SUPPRESS WAWRNING
import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
#=================================================================

SEED = 10
from numpy import random
random.seed(SEED)

root = 'datasets/EuroSAT_processed_v2_CNN_0'

#=================================================================
# Load Data
images, labels = [],[]
classes = glob.glob(os.path.join(root, '*'))
for i, c in enumerate(tqdm(classes)):
    
    for k, img in enumerate(glob.glob(os.path.join(c, '*'))):
        
        lbl = np.zeros((len(classes)))
        lbl[i] = 1
        
        labels.append(lbl)
        images.append(np.load(img))
ccl = []

for c in classes:
    ccl.append(c.split(os.sep)[-1])

images = np.array(images)
labels = np.array(labels)

#=================================================================
# Train Validation Split
rdn_idx = np.random.choice(np.arange(images.shape[0]), size = images.shape[0], replace=False)
train_s = int(len(rdn_idx)*0.8)
train_idx = rdn_idx[:train_s]
valid_idx = rdn_idx[train_s:]

x_train = images[train_idx,...]
y_train = labels[train_idx,...]
x_test  = images[valid_idx,...]
y_test  = labels[valid_idx,...]

#=================================================================
# AutoDL
input_node = ak.ImageInput()
output_node = ak.ImageBlock(
    block_type = 'vanilla',
    normalize  = False,
    augment = False,
)(input_node)
output_node = ak.ClassificationHead()(output_node)

clf = ak.AutoModel(
                    inputs = input_node, 
                    outputs = output_node, 
                    overwrite = False, 
                    max_trials = 1, 
                    loss='categorical_crossentropy', 
                    metrics=['accuracy'], 
                    objective='val_accuracy',
                    directory = os.path.join('results', root.split(os.sep)[-1])
                    )

clf.fit(x_train, y_train, validation_data=(x_test, y_test), epochs = 100, batch_size = 16)
y_pred = clf.predict(x_test, verbose = 0)

cmtx = pd.DataFrame(
    confusion_matrix(np.argmax(y_test, axis = -1), np.argmax(y_pred, axis = -1), normalize='true'), 
    index=['{}'.format(x) for x in ccl], 
    columns=['{}'.format(x) for x in ccl]
)

print('')
print(cmtx)
print('')
print(classification_report(y_test, y_pred, target_names=ccl))