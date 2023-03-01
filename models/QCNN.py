from tensorflow.keras.layers import Input, Activation, Flatten, Dense, Conv2D, Dropout, AveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

from data.datareader import datareader
from .qconfig import *

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm.auto import tqdm
from io import StringIO
import pandas as pd
import numpy as np
import os

class QCNNv1:
    '''
        Quantum Convolutional Neural Network Class (QCNN) Version 1. This class is ment to be used to
        develop hybrid quantum convolutional neural networks. This QCNN works with features maps obtained
        with a quantum cicuit, indeed it is composed of only dense layers, since the convolution operator has
        been already applied.
    '''

    def __init__(self, img_shape, n_classes, name=None):
        '''
            Class builder.

            Inputs:
                - img_shape: shape of the feature maps, channel last
                - n_classes: number of output classes
                - name: name of the QCNN, default QCNNv1
        '''

        self.img_shape = img_shape
        self.n_classes = n_classes
        self.name = name
        if self.name == None: self.name = 'QCNNv1'

        # Model Settings
        self.loss          = qcnnv1s['loss']
        self.metrics       = qcnnv1s['metrics']
        self.learning_rate = qcnnv1s['learning_rate']
        self.dropout       = qcnnv1s['dropout']
        self.batch_size    = qcnnv1s['batch_size']
        self.epochs        = qcnnv1s['epochs']
        self.es_rounds     = qcnnv1s['early_stopping']
        self.dense         = qcnnv1s['dense']
        self.conv          = qcnnv1s['conv']
        self.kernel        = qcnnv1s['kernel']
        self.stride        = qcnnv1s['stride']
        self.pool_size     = qcnnv1s['pool_size']
        self.pool_stride   = qcnnv1s['pool_stride']

        self.model = self.__build() 

    def __build(self):
        '''
            This method builds the QCNN.
        '''

        xin = Input(shape=self.img_shape)
        x   = Activation('relu')(xin)
        x   = AveragePooling2D(pool_size = self.pool_size, strides = self.pool_stride)(x)
        
        if self.conv is not None:
            # Convolutional Layers
            for conv in self.conv:
                x   = Conv2D(filters = conv, kernel_size = self.kernel, strides = self.stride, activation='relu')(x)
                x   = AveragePooling2D(pool_size = self.pool_size, strides = self.pool_stride)(x)
        
        x   = Flatten()(x)
        
        # Dense Layers
        for dense in self.dense:
            x   = Dropout(self.dropout)(x)
            x   = Dense(dense, activation='relu')(x)

        # Final Dense Layer
        x   = Dropout(self.dropout)(x) 
        x   = Dense(self.n_classes, activation='softmax')(x)
        
        model = Model(inputs=xin, outputs=x, name=self.name)
        model.compile(optimizer = Adam(learning_rate=self.learning_rate),
                      loss      = self.loss,
                      metrics   = self.metrics)

        return model
    
    def train_test(self, train_dataset, val_dataset, labels_mapper, normalize=None, verbose=0):
        '''
            Train and test the model.

            The model is firstly trained, then the weights and the training history is saved.
            Afterwards the model is tested on both the training and the validation set. Also
            in this case resutls are saved.

            Inpus:
             - train_dataset: tuple containing training paths and labels
             - val_dataset:   tuple containing validation paths and laels
             - labels_mapper: dictionary mapping the predicted classes (one hot encoded) into
                              class names
             - normalize:     if true apply normalization (used by data loaders)
             - verbose:       verbose for training (0 silent, 1 progressbar, 2 loss)
        '''

        # Early Stopping to avoid overfitting
        es = EarlyStopping(monitor='val_loss', 
                           patience=self.es_rounds,
                           mode='auto',
                           verbose=0,
                           baseline=None)
        
        # Training and Validation data loader
        train_gen = datareader.generator(train_dataset, 
                                         self.batch_size,
                                         self.img_shape,
                                         normalize=normalize)
        val_gen   = datareader.generator(val_dataset,
                                         self.batch_size,
                                         self.img_shape,
                                         normalize=normalize)
        # Model Training
        history = self.model.fit(
                train_gen,
                steps_per_epoch  = len(train_dataset[0])//self.batch_size,
                validation_data  = val_gen,
                validation_steps = len(val_dataset[0])//self.batch_size,
                epochs           = self.epochs,
                callbacks        = [es],
                verbose          = verbose)
    
        self.history = history
        
        # Make dir to save results
        path = os.path.join('results', 'QCNNv1',
                            '{}'.format(datetime.now().strftime("%d-%m-%Y-%H:%M:%S")))
        os.makedirs(path)

        # Save model
        model_path = os.path.join(path, 'model.h5')
        self.model.save(model_path)
        print('{:<30s}{}'.format('Model Saved', model_path))
        # Save history
        history_path = os.path.join(path,'history.csv')
        df = pd.DataFrame(history.history)
        df.to_csv(history_path, index=False)
        print('{:<30s}{}'.format('History Saved', history_path))
        # Save model settings
        self.__save_model_settings(path) 
        # Test model
        self.test(train_dataset, val_dataset, path, labels_mapper, normalize)
    
    def __save_model_settings(self, path):
        '''
            Saves Model settings and summary to a file.

            Input:
                - path: path to save the file
        '''
        
        settings_path = os.path.join(path, 'settings.txt')
        with open(settings_path, 'w') as f:
            f.write('{:.^100}\n'.format('Model Settings'))
            f.write('{:<30s}:{}\n'.format('Name',                    self.name))  
            f.write('{:<30s}:{}\n'.format('Image Shape',             self.img_shape))

            # Write Config file to settings.txt
            for key, value in qcnnv1s.items():
                f.write('{:<30s}:{}\n'.format(key, value))          

            tmp_smry = StringIO()
            self.model.summary(print_fn=lambda x: tmp_smry.write(x + '\n'))
            summary = tmp_smry.getvalue()
            f.write('{:.^100}\n'.format('Model Parameters'))
            f.write(summary)

        print('{:<30s}{}'.format('Model Settings Saved',settings_path))
    
    def test(self, train_dataset, val_dataset, path, labels_mapper, normalize=None):
        '''
            Test the model and save results.

            Inputs:
                - train_dataset: tuple containing training paths and labels
                - val_dataset:   tuple containing validation paths and laels
                - labels_mapper: dictionary mapping the predicted classes (one hot encoded) into
                                 class names
                - path:          path to save results
                - normalize:     if true apply normalization (used by data loaders)
        '''

        # Training and Validation data loader
        train_gen = iter(datareader.generatorv2(train_dataset, 
                                         self.img_shape,
                                         normalize=normalize))
        val_gen   = iter(datareader.generatorv2(val_dataset,
                                         self.img_shape,
                                         normalize=normalize))

        # Training Results
        print('Testing model on training set')
        self.__make_pred(train_dataset, train_gen, path, 'training', labels_mapper)
        print('Testing model on valdiation set')
        self.__make_pred(val_dataset,   val_gen,   path, 'validation', labels_mapper)

    def __make_pred(self, dataset, iterator, path, name, labels_mapper):
        '''
            Make model prediction over a dataset and save results

            Inputs:
                - dataset:       tuple containing paths and labels
                - iterator:      data loader as iterator
                - path:          path to save results
                - name:          name of the dataset
                - labels_mapper: dictionary mapping the predicted classes (one hot encoded) into
                                 class names
        '''

        predictions = np.zeros(np.shape(dataset[1]))
        targets     = np.zeros(np.shape(dataset[1]))
        paths       = []

        # Iterate through the dataset (can be training or validation)
        for i in tqdm(range(len(dataset[0]))):
            x, y, ps = next(iterator)
            p = self.model.predict(x[np.newaxis,...], verbose = 0)
            predictions[i] = p[0]
            targets[i]     = y
            paths.append(ps)
        
        training_res = os.path.join(path, name+'-results.csv')
        df = pd.DataFrame({'Classes'    : list(map(labels_mapper.get, list(np.argmax(targets, axis=-1)))),
                           'Predictions': np.argmax(predictions, axis=-1), 
                           'Targets':     np.argmax(targets, axis=-1),
                           'Paths':       paths})
        df.to_csv(training_res, index=False)
        print('{:<30s}{}'.format(name +' Results', training_res)) 
        self.__confusion_matrix_report(path, name, np.argmax(targets, axis=-1), np.argmax(predictions, axis=-1), labels_mapper.values())

    def __confusion_matrix_report(self, path, name, targets, predictions, classes, display = False):
        '''
            Plot the confusion matrix and compute the classification report.

            Inputs:
                - path:        path to save confusion matrix and classification report
                - name:        name of the dataset
                - targets:     image labels
                - predictions: model predictions
                - classes:     name of the classes
                - display:     if true plot the confusion matrix
        '''

        # Plot the Confusion Matrix
        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(10,8))
        cm = confusion_matrix(targets, predictions, normalize='true')
        cmd = ConfusionMatrixDisplay(cm, display_labels=classes)
        cmd.plot(ax = ax, xticks_rotation=90, cmap = 'Blues', values_format='.2f')
        cmd.ax_.get_images()[0].set_clim(0, 1)
        
        fig.tight_layout()
        if display: plt.show()

        cf_path = os.path.join(path, name+'-cf.png')
        fig.savefig(cf_path)
        print('{:<30s}{}'.format('Confusion matrix saved', cf_path))
        plt.close()
        
        # Calculate the Classification Report
        c_report = classification_report(targets, predictions, target_names=classes)
        
        report_path = os.path.join(path, name+'-report.txt')
        with open(report_path, 'w') as f:
            f.write(c_report)
            
        print('{:<30s}{}'.format('Classification report saved', report_path))
