from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from .qconfig import *

from data.datareader import datareader


class QCNNv1:

    def __init__(self, img_shape, n_classes, name=None):
        
        self.img_shape = img_shape
        self.n_classes = n_classes
        self.name = name
        if self.name == None: self.name = 'QCNNv1'
        self.model = self.__build() 

    def __build(self):
        
        xin = Input(shape=self.img_shape)
        x   = Activation('relu')(xin)
        x   = MaxPooling2D(3)(x)
        x   = Flatten()(x)
        x   = Dropout(qcnnv1s['dropout'])(x)
        x   = Dense(128, activation='relu')(x)
        x   = Dropout(qcnnv1s['dropout'])(x)
        x   = Dense(self.n_classes, activation='softmax')(x)
        
        model = Model(inputs=xin, outputs=x, name=self.name)
        model.compile(optimizer = Adam(learning_rate=qcnnv1s['learning_rate']),
                      loss      = qcnnv1s['loss'],
                      metrics   = qcnnv1s['metrics'])

        return model
    
    def train(self, train_dataset, val_dataset):

        es = EarlyStopping(monitor='val_loss', 
                           patience=qcnnv1s['early_stopping'],
                           mode='auto',
                           verbose=0,
                           baseline=None)

        train_gen = datareader.generator(train_dataset, 
                                         qcnnv1s['batch_size'],
                                         self.img_shape,
                                         normalize=None)
        val_gen   = datareader.generator(val_dataset,
                                         qcnnv1s['batch_size'],
                                         self.img_shape,
                                         normalize=None)

        history = self.model.fit(
                train_gen,
                steps_per_epoch  = len(train_dataset[0])//qcnnv1s['batch_size'],
                validation_data  = val_gen,
                validation_steps = len(val_dataset[0])//qcnnv1s['batch_size'],
                epochs           = qcnnv1s['epochs'],
                callbacks        = [es])

        self.history = history






