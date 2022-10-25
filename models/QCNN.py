from tensorflow.keras.layers import Input, Activation, Flatten, Dense, Conv2D, MaxPooling2D, Dropout, AveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

from data.datareader import datareader
from .qconfig import *

from datetime import datetime
from tqdm.auto import tqdm
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
        self.loss         = qcnnv1s['loss']
        self.metrics      = qcnnv1s['metrics']
        self.learning_rate = qcnnv1s['learning_rate']
        self.dropout      = qcnnv1s['dropout']
        self.batch_size   = qcnnv1s['batch_size']
        self.epochs       = qcnnv1s['epochs']
        self.es_rounds    = qcnnv1s['early_stopping']

        self.model = self.__build() 

    def __build(self):
        '''
            This method builds the QCNN.
        '''
        xin = Input(shape=self.img_shape)
        x   = Activation('relu')(xin)
        x   = AveragePooling2D(pool_size = 2, strides = 2)(x)
        x   = Flatten()(x)
        x   = Dropout(self.dropout)(x)
        x   = Dense(128, activation='relu')(x)
        x   = Dropout(self.dropout)(x)
        x   = Dense(64,  activation='relu')(x)
        x   = Dropout(self.dropout)(x) 
        x   = Dense(self.n_classes, activation='softmax')(x)
        
        model = Model(inputs=xin, outputs=x, name=self.name)
        model.compile(optimizer = Adam(learning_rate=self.learning_rate),
                      loss      = self.loss,
                      metrics   = self.metrics)

        return model
    
    def train_test(self, train_dataset, val_dataset, labels_mapper, normalize=None):
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
                callbacks        = [es])
    
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
        

        from io import StringIO
        settings_path = os.path.join(path, 'settings.txt')
        with open(settings_path, 'w') as f:
            f.write('{:.^100}\n'.format('Model Settings'))
            f.write('{:<30s}:{}\n'.format('Name',                  self.name))  
            f.write('{:<30s}:{}\n'.format('Image Shape',           self.img_shape))  
            f.write('{:<30s}:{}\n'.format('Loss Function',         self.loss))          
            f.write('{:<30s}:{}\n'.format('Evaluation Metrics',    self.metrics))      
            f.write('{:<30s}:{}\n'.format('Learning Rate',         self.learning_rate)) 
            f.write('{:<30s}:{}\n'.format('Dropout',               self.dropout))
            f.write('{:<30s}:{}\n'.format('Batch Size',            self.batch_size))   
            f.write('{:<30s}:{}\n'.format('Epochs',                self.epochs))
            f.write('{:<30s}:{}\n'.format('Early Stopping Rounds', self.es_rounds))    
            

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
        self.__make_pred(train_dataset, train_gen, path, 'training', labels_mapper)
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




