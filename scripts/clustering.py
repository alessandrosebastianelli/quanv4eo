from data.datahandler import datahandler
from data.datareader import datareader
from utils.plotter import *

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import Birch
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from numpy import unique
from numpy import where
import numpy as np
import os

import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

SEED = 10
from numpy import random
random.seed(SEED)

dataset_name = 'EuroSAT_processed_QCNN_1'
root = os.path.join('datasets', dataset_name)
dhandler = datahandler(root)
dhandler.print_report(name=dataset_name)
labels_mapper, x, y = dhandler.unpack(dhandler.paths)
print('Labels')
for key in labels_mapper: print('{:<30s}{}'.format(key,labels_mapper[key]))
print('\nDataset Size')
print('{:<30s}{}'.format('Images', len(x)))
print('\nTraining Dataset samples')
print('{:<30s}{:<80s}{:<10s}{}'.format('X Train', x[0], 'Size', np.load(x[0]).shape))
print('{:<30s}{}'.format('X Train', y[0]))
classes = list(labels_mapper.keys())

def reshape(x):
    for i in range(x.shape[-1]):
        if i == 0:
            vals = x[:,:,i].flatten()
        else:
            vals = np.concatenate((vals, x[:,:,i].flatten()))
    return vals

loader  = iter(datareader.generatorv2((x, y), (3,3,12)))
X = []
Y = []

for _ in tqdm(range(len(x))):    
    it = next(loader)
    lbl = np.argmax(it[1])
    #if lbl == 0 or lbl == 1:
    X.append(reshape(it[0]))
    Y.append(lbl)

Y = np.array(Y)
X = np.array(X)



scaler = StandardScaler()
X_std = scaler.fit_transform(X)

pca = PCA(n_components=10, random_state=SEED)#, svd_solver='full')
X_std = pca.fit_transform(X_std)
print(pca.explained_variance_ratio_)

N_CLUSTERS = 10

errors, threshold, branching_factor = [], [], []
best_err, best_th, best_br = 1000, 0, 0

for t in np.flip(np.arange(0.01, 1.01, 0.01)):
    for b in range(2, 100, 1):
        km = Birch(threshold=t, #0.86, #1, 
                    branching_factor=b, #9,#5,
                    n_clusters=N_CLUSTERS,
                    compute_labels=True,
                    copy=True)

        yhat = km.fit_predict(X_std)
        yhatct = -np.sort(-np.bincount(yhat))
        yct = -np.sort(-np.bincount(Y))
        err = np.sum(np.abs(yct - yhatct))/ np.sum(yct)   
        
        if err <= best_err:
            best_err=err
            best_th=t
            best_br=b
        
        print('\r Current Error {:.5f} - Threshold {:.2f} - Branching Factor {} -- Best Error {:.5f} - Threshold {:.2f} - Branching Factor {} '.format(err,t,b,best_err, best_th, best_br), end='\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t')
        errors.append(err)
        threshold.append(t)
        branching_factor.append(b)

M = np.argmin(errors)
print('Min {} - Error - {:.5f} - Threshold {:.2f} - Branching Factor {}'.format(M, errors[M], threshold[M], branching_factor[M]))

M = np.argmin(errors)
print('Min {} - Error - {:.5f} - Threshold {:.2f} - Branching Factor {}'.format(M, errors[M], threshold[M], branching_factor[M]))
km = Birch(threshold=threshold[M], #0.86, #1, 
                    branching_factor=branching_factor[M], #9,#5,
                    n_clusters=N_CLUSTERS,
                    compute_labels=True,
                    copy=True)

yhat = km.fit_predict(X_std)
yhatct = -np.sort(-np.bincount(yhat))
yct = -np.sort(-np.bincount(Y))
err = np.sum(np.abs(yct - yhatct))/ np.sum(yct)
print('Recomputed Error {:.5f}'.format(err))

fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (25, 5), gridspec_kw={'width_ratios': [3, 2]})
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for i, cluster in enumerate(clusters):
    # get row indexes for samples with this cluster
    row_ix = where(yhat == cluster)
    # create scatter of these samples
    ax[0].scatter(X_std[row_ix, 0], X_std[row_ix, 1], marker='+', label=classes[i])
    ax[0].set_xlabel('PCA Component 1')
    ax[0].set_ylabel('PCA Component 2')

ax[0].legend(loc='lower right')

xx = np.arange(N_CLUSTERS) - 0.2
ax[1].bar(xx,  yhatct, width=0.3, align='center', label='Birch', color='purple')

for i, val in enumerate(yhatct):
    ax[1].text(i-0.5, 20+val, val, fontsize=8, rotation=30)

xx = np.arange(N_CLUSTERS) + 0.2
ax[1].bar(xx,  yct, width=0.3, align='center', label='Ground Truth', color='brown')
for i, val in enumerate(yct):
    ax[1].text(i, 20+val, val, fontsize=8, rotation=30)

ax[1].set(xticks=range(N_CLUSTERS), xlim=[-1, N_CLUSTERS])    
ax[1].set_xticklabels(classes, rotation=45, ha='right')
ax[1].set_ylabel('#images')

mm = np.bincount(yhat).max()
ax[1].set_ylim([None,mm+(20*mm)/100])
ax[1].legend(loc='lower left')

plt.show()
plt.close()