import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np

def dlv2(dataset, loader, classes):
    '''
        Test a data loader by loading all the images in the dataset, both to measure the time
        and also to verify if all the images are correctly loaded. Produces a plot with the 
        statistics.

        Inputs:
            - dataset: tuple containing images path and labels
            - loader:  data loader to be tested
            - classes: name of the classes, it is used for plotting reasons
            
    '''
    # Transform the data loader into an iterator
    loader = iter(loader)
    # Unpack the dataset
    x = dataset[0] # Inputs
    y = dataset[1] # Labels
    
    # Count will collect the number of samples for each class
    count = np.zeros(len(y[0]))
    
    # Load dataset
    for i in tqdm(range(len(x)), desc='Reading Image', colour='black'):
        (xi,yi,pi) = next(loader)
        count[np.argmax(yi)] += 1
    
    # Barh plot for variable Count
    yy = len(y[0])
    xx = np.arange(yy)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(2*yy/3, 4))
    ax.barh(xx, count, color='black')
    for i in range(len(count)):
        ax.text(x=count.mean()/3, y=i, s=count[i], c='white', verticalalignment='center')
    ax.set_yticks(xx)
    ax.set_title('Testing Generator - Dataset Statistics')
    ax.set_xlabel('# of images')
    ax.set_ylabel('Class')
    ax.set_yticklabels(labels = classes)
    fig.tight_layout()
    plt.show()
    plt.close()
