import matplotlib.pyplot as plt

def plot_result(img, out):
    fig, axes = plt.subplots(nrows = 2, ncols = out.shape[-1]+1, figsize = (2*(out.shape[-1]+1),4))
    axes[0,0].imshow(img, vmin = 0, vmax = 1)
    axes[0,0].axis('off')
    axes[0,0].set_title('Input Image')

    axes[1,0].hist(img[...,0].flatten(), 60, color='red')
    axes[1,0].hist(img[...,1].flatten(), 60, color='green')
    axes[1,0].hist(img[...,2].flatten(), 60, color='blue')

    for i in range(out.shape[-1]):
        axes[0,i+1].imshow(out[...,i], vmin = -1, vmax = 1)
        axes[0,i+1].set_title('QCNN - F. Map {}'.format(i))
        axes[0,i+1].axis('off')
        
        axes[1,i+1].hist(out[...,i].flatten(), 60, color='black')
        axes[1,i+1].set_xlim([-1,1])

    fig.tight_layout()
    plt.show()
    plt.close()

def plot_features_map(feat_maps):
    fig, axes = plt.subplots(nrows = 2, ncols = feat_maps.shape[-1], figsize = (2*(feat_maps.shape[-1]+1),4))
    
    for i in range(feat_maps.shape[-1]):
        axes[0,i].imshow(feat_maps[...,i], vmin = -1, vmax = 1)
        axes[0,i].set_title('QCNN - F. Map {}'.format(i))
        axes[0,i].axis('off')
        
        axes[1,i].hist(feat_maps[...,i].flatten(), 60, color='black')
        axes[1,i].set_xlim([-1,1])

    fig.tight_layout()
    plt.show()
    plt.close()