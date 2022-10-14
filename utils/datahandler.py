import glob
import os
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from qlayers import *
import cv2
from joblib import Parallel, delayed
import tensorflow as tf

class datahandler:
    def __init__(self, root):
        self.root = root
        classes, self.paths, self.labels = self.__load_paths()

        self.classes = []
        for c in classes:
            self.classes.append(c.split(os.sep)[-1])

    def __load_paths(self):
        classes = glob.glob(os.path.join(self.root, '*'))
        label = np.zeros((len(classes)))

        labels = []
        paths = []
        
        for i, c in tqdm(enumerate(classes)):
            imgs_c = glob.glob(os.path.join(c, '*'))
            for img in imgs_c:
                paths.append(img)
                label[i] = 1
                labels.append(label.tolist())
                label = np.zeros((len(classes)))
        
        z = list(zip(paths, labels))
        random.shuffle(z)
        paths, labels = zip(*z)
        
        paths = np.array(paths)
        labels = np.array(labels)

        return classes, paths, labels


    def __shuffle_dataset(self, dataset):
        z = list(zip(dataset[0], dataset[1]))
        random.shuffle(z)
        paths, labels = zip(*z)
        paths = np.array(paths)
        labels = np.array(labels)

        return (paths, labels)

    def split(self, split_factor = 0.2):
        val_size = int(len(self.paths)*split_factor)
        train_size = int(len(self.paths) - val_size)
        
        return (self.paths[0:train_size], self.labels[0:train_size, ...]), (self.paths[train_size:train_size+val_size], self.labels[train_size:train_size+val_size, ...])

                    