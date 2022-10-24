import numpy as np

def convert_labels_mapper(old_dict):
    '''
        It converts a class dictionary 
        from 
            key: class name -> item: one hot encoded label 
        into
            key: argmax(one hot encoded label) -> item: class name

        It is mainly used by object in folder models to save results pandas dataframe
    '''

    new_dict = {}
    for x, y in old_dict.items():
        new_dict[np.argmax(y)] = x

    return new_dict
