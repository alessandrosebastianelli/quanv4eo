import matplotlib.pyplot as plt
import numpy as np
import rasterio
import random

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
        z = list(zip(dataset[0], dataset[1]))
        random.shuffle(z)
        paths, labels = zip(*z)
        paths = np.array(paths)
        labels = np.array(labels)

        return (paths, labels)

    @staticmethod
    def generator(dataset, batch_size, img_shape):

        dataset = datareader.__shuffle(dataset)

        paths  = dataset[0]
        labels = dataset[1]

        x_in  = np.zeros((batch_size, ) + img_shape)
        x_out = np.zeros((batch_size, ) + labels[0].shape)
        
        print(x_in.shape, x_out.shape)
        
        while False:

            for i in range(batch_size):
                pass


        

        