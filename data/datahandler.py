import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import random
import numpy as np
import glob
import os

class datahandler:
    '''
        Data Handler, it handle the dataset
    '''

    def __init__(self, root):
        '''
            It creates a Data Handler object.
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            The dataset must be organized in this format:

            QuantumCNN
            │   README.md
            │   requirements.txt    
            │
            └───circuits
            └───...
            └───datasets
                └───EuroSAT
                    └───Highway
                            highway1.jpg
                            highway2.jpg                
                    └─── ....
                    └───Lake
                            lake1.jpg
                            lake2.jpg 

            The following image format are supported:
                - .png
                - .jpg
                - .jpeg
                - .tif
                - .npy
            please adapt your format accordingly. 

            Images should contain all the bands in a single file.
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            Input
                - root: root path of the dataset
        '''

        self.root = root
        self.paths = self.__load_paths()

    def __load_paths(self, verbose=False):
        '''
            Loads the images path

            Output:
                - paths: python dictionary, keys==classes, values==paths
        '''

        classes_raw = glob.glob(os.path.join(self.root, '*'))

        paths = {}
        for i, c in tqdm(enumerate(classes_raw), disable=not(verbose), colour='black'):
            imgs_c = glob.glob(os.path.join(c, '*'))
            paths[c.split(os.sep)[-1]] = imgs_c

        return paths

    def print_report(self, paths=None, name=None):
        '''
            Print dataset statistics

            Inputs:            
            - paths: python dictionary, keys==classes, values==paths (can be applied to dataset splits)
                    If None, wil print the whole dataset statistics
            - name: name of the dataset to be printed
        '''
        
        if paths == None: paths = self.paths   
        if name  == None: name = "No name"
        
        print('{:=^100}\n'.format(''))
        print('Dataset {}\n'.format(name))
        for i, c in enumerate(paths.keys()): print('Class {} - {:<25s} - #images: {}'.format(i,c,len(paths[c])))

    def split(self, paths, factor=0.2):
        '''
            Split the dataset given a split facotor

            Inputs:            
                - paths: python dictionary, keys==classes, values==paths (can be applied to dataset splits)
                        If None, wil use whole dataset
                - facotr: split factor, must range between 0 and 1
            Outputs:
                - split_A: python dictionary, keys==classes, values==paths. 
                           It collects the (1-factor)*100 % of the data
                - split_B: python dictionary, keys==classes, values==paths. 
                           It collects the (factor)*100 % of the data
                           
        '''

        if paths == None: paths = self.paths
        if factor < 0: factor=0
        if factor > 1: factor=1

        split_A, split_B = {}, {}
    
        for i, c in enumerate(paths.keys()):
            # Calculates the split factor for each class indipendentely
            l = len(paths[c])
            val_size   = int(l*factor)
            train_size = int(l - val_size)

            # Apply split and fill relative dictionary
            split_A[c] = paths[c][:train_size]
            split_B[c] = paths[c][train_size:]

        return split_A, split_B

    def unpack(self, paths, shuffle=True):
        '''
            Unpack paths into an array and shuffles it

            Inputs:
                - paths: python dictionary to be unpacked, keys==classes, values==paths 
                        (can be applied to dataset splits)
                - shuffle: if True applies a random shuffle to the unpacked array
            Output:
                - unpacked_paths: 1D array containing unpacked paths
                - unpacked_labels: one-hot encoded class for each path
                - labels mapper: dictionary that maps the class name with the one-hot encoded vector
        '''
    
        labels_mapper = {}
        for i, c in enumerate(paths.keys()):
            x = np.zeros((len(paths.keys())))
            x[i] = 1
            labels_mapper[c] = x

        unpacked_paths = []
        unpacked_labels = []
        for i, c in enumerate(paths.keys()):
            for p in paths[c]:
                unpacked_paths.append(p)
                unpacked_labels.append(labels_mapper[c])
        
        return labels_mapper, unpacked_paths, unpacked_labels
