import numpy as np

def convert_labels_mapper(old_dict):

    new_dict = {}
    for x, y in old_dict.items():
        new_dict[np.argmax(y)] = x

    return new_dict