import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import os


def plot_result(img, out):
    '''
        Plot the input image for the Quantum Convolution and relative output (feature map).

        Inputs:
            - img: the input image (channel last) for the quantum convolution
            - out: the output feature maps (channel last) from the quantum convolution
    '''

    fig, axes = plt.subplots(nrows = 2, ncols = out.shape[-1]+1, figsize = (2*(out.shape[-1]+1),4))
    axes[0,0].imshow(img, vmin = 0, vmax = 1)
    axes[0,0].axis('off')
    axes[0,0].set_title('Input Image')

    axes[1,0].hist(img[...,0].flatten(), 60, color='red')
    axes[1,0].hist(img[...,1].flatten(), 60, color='green')
    axes[1,0].hist(img[...,2].flatten(), 60, color='blue')

    for i in range(out.shape[-1]):
        axes[0,i+1].imshow(out[...,i], vmin = -1, vmax = 1)
        axes[0,i+1].set_title('QCNN - F. Map {}'.format(i))
        axes[0,i+1].axis('off')
        
        axes[1,i+1].hist(out[...,i].flatten(), 60, color='black')
        axes[1,i+1].set_xlim([-1,1])

    fig.tight_layout()
    plt.show()
    plt.close()

def plot_features_map(feat_maps):
    '''
        Plot the feature maps matrix in a n cols plot, where n depends on the number of
        the feature map in feat_maps.
        
        Inputs:
            feat_maps: channel last array with N channels

    '''

    fig, axes = plt.subplots(nrows = 2, ncols = feat_maps.shape[-1], figsize = (2*(feat_maps.shape[-1]+1), 4))
    
    for i in range(feat_maps.shape[-1]):
        axes[0,i].imshow(feat_maps[...,i], vmin = -1, vmax = 1)
        axes[0,i].set_title('QCNN - F. Map {}'.format(i))
        axes[0,i].axis('off')
        
        axes[1,i].hist(feat_maps[...,i].flatten(), 60, color='black')
        axes[1,i].set_xlim([-1,1])

    fig.tight_layout()
    plt.show()
    plt.close()

def plot_training(name, display = True, latest=False):
    '''
        This function plots the training and validation curves of a specific model, merging
        all the simulations done with that model, unless latest is set to True.

        Inputs:
            - name:     name of the model (must correspond to a real name used in the main files,
                        a folder with the same name must be in results folder)
            - display:  if True it displays the plot, in any case a copy is saved in 
                        results/name/training.png
            - latest:   if true plot the training and validation curves for only the last model
                        simulation
    '''

    path = os.path.join('results', name)
    results = glob.glob(os.path.join(path, '*'+os.sep))
    
    results.sort()

    if latest: results = results[-1]

    df = pd.read_csv(os.path.join(results[0], 'history.csv'))
    nc = len(df.columns)
    
    size = 10*np.log(1+len(df[df.columns[0]]))
    # Plot history
    fig, ax = plt.subplots(nrows=nc, ncols=1, figsize=(size, size//nc))
    for result in results:
        df = pd.read_csv(os.path.join(result, 'history.csv')) 
        for n in range(nc):
            df[df.columns[n]].plot(ax=ax[n], style='.-', label = result.split(os.sep)[-2])
            ax[n].set_title(df.columns[n])
            ax[n].legend()
            ax[n].set_xlabel('Epochs')
            ax[n].set_xticks(np.arange(len(df[df.columns[n]]))) 

    fig.tight_layout()
    if display: plt.show()

    fig.savefig(os.path.join('results', name, 'training.png'))
    print('Image saved')

    plt.close()
