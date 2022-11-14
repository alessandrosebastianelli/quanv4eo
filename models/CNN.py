from tensorflow.keras.layers import Input, Activation, Flatten, Dense, Conv2D, MaxPooling2D, Dropout, AveragePooling2D
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
import matplotlib
import os


class CNN:
    

    def __init__(self):
        pass

    def __build(self):
        pass

    

