qcnnv1s = {
            'loss':            'categorical_crossentropy',
            'learning_rate':   0.0002,
            'metrics':         ['accuracy', 'mse'],
            'dropout':         0.2,
            'batch_size':      16,
            'epochs':          100,
            'early_stopping':  10,
            'dense':           [256, 128, 64],                      # Vector of #neurons for each dense layer
            'conv':            [128],                               # Vector of #filters for each convolution layer
            'kernel':          3,                                   # Kernel Size for the convolution
            'stride':          1,                                   # Strides for the convolution
            'pool_size':       2,                                   # Kernel Size for the Global Average Pooling
            'pool_stride':     2,                                   # Strides for the Global Average Pooling
        }