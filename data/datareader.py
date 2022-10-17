from .normalizer import normalizer
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import random
import cv2

class datareader:
    '''
        Class containing static methods for reading images
    '''

    @staticmethod
    def load(path):
        '''
            Load an image and its metadata given its path.
            
            The following image format are supported:
                - .png
                - .jpg
                - .jpeg
                - .tif
                - .npy
            please adapt your format accordingly. 
            
            Inputs:
                - path: position of the image, if None the function will ask for the image path using a menu
                - info (optional): allows to print process informations
            Outputs:
                - data: WxHxB image, with W width, H height and B bands
                - metadata: dictionary containing image metadata
        '''
        
        MPL_FORMAT = ['.png', '.jpg', '.jpeg']
        RIO_FORMAT = ['.tif', '.tiff']
        NP_FORMAT  = ['.npy']
        
        if any(frmt in path for frmt in RIO_FORMAT):
            with rasterio.open(path) as src:
                data     = src.read()
                metadata = src.profile
            data = np.moveaxis(data, 0, -1)
            
        elif any(frmt in path for frmt in MPL_FORMAT):
            data = plt.imread(path)
            metadata = None

        elif any(frmt in path for frmt in NP_FORMAT):
            data     = np.load(path)
            metadata = None
            
        else:
            data     = None
            metadata = None
            print('!!! File can not be opened, format not supported !!!')
            
        return data, metadata

    @staticmethod
    def __shuffle(dataset):
        '''
            Random shuffle the dataset

            Inputs:
                - dataset: tuple of image paths and labels. (paths, labels)
            Output:
                - dataset: tuple of shuffled image paths and labels. (paths, labels)

        '''
        z = list(zip(dataset[0], dataset[1]))
        random.shuffle(z)
        paths, labels = zip(*z)
        paths = np.array(paths)
        labels = np.array(labels)

        return (paths, labels)

    @staticmethod
    def generator(dataset, batch_size, img_shape, normalize = True):
        '''
            Basic Keras-like image loader

            Inputs:
                - dataset: tuple of image paths and labels. (paths, labels)
                - batch_size: int value represent the batch of image to be loaded iteratively
                - img_shape: shape of the image that need to be loaded
                - normalize: if true apply normalization
            Yield:
                - x_in: tensor containg a batch size of images
                - x_ou: tensor containg a batch size of labels
        '''
        dataset = datareader.__shuffle(dataset)
        paths  = dataset[0]
        labels = dataset[1]

        counter = 0
        while True:
            x_in = np.zeros((batch_size, ) + img_shape)
            x_ou = np.zeros((batch_size, ) + labels[0].shape)

            for i in range(batch_size):
                # Load an image and a label
                img, _ = datareader.load(paths[counter])
                lbl    = labels[counter]
                # If normalizer is try apply normalization, in this case I put minmax,
                # you can decide among the ones available in normalizer class
                if normalize != None: img = normalizer.minmax_scaler(img)
                # Apply reshape if there is a mistmach between img_shape and real image shape
                if img_shape != img.shape: img = cv2.resize(img, shape)
                # Fill the batch input and output vectors
                x_in[i, ...] = img
                x_ou[i, ...] = lbl
            
                counter += 1
            # If the counter is greater than the dataset size, reset counter and shuffle the dataset
            if counter >= len(paths) - batch_size:
                dataset = datareader.__shuffle(dataset) 
                counter = 0

            yield x_in, x_ou


        

        