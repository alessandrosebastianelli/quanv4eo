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
        axes[0,i+1].set_title('QCNN - B {}'.format(i))
        axes[0,i+1].axis('off')
        
        axes[1,i+1].hist(out[...,i].flatten(), 60, color='black')

    fig.tight_layout()
    plt.show()
    plt.close()