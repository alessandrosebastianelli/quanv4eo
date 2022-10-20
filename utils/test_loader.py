import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np

def dlv2(dataset, loader, classes):
    loader = iter(loader)
    x = dataset[0]
    y = dataset[1]

    count = np.zeros(len(y[0]))
    
    for i in tqdm(range(len(x)), desc='Reading Image', colour='black'):
        (xi,yi,pi) = next(loader)
        count[np.argmax(yi)] += 1

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