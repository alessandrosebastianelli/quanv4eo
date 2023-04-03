from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


df = pd.read_csv('results.csv')
q  = df.qubits
f  = df.filters
k  = df.kernelsize.values.astype(np.int32)
st = df.stride.values.astype(np.int32)
s  = df.imgsize.values.astype(np.int32)
t  = df.time

fig= plt.figure(figsize = (16,4))

ax1 = fig.add_subplot(131, projection = '3d')
img = ax1.scatter(s,q,t, c=st, cmap='viridis', vmin=1, vmax=3)
ax1.set_xlabel('Image Size')
ax1.set_ylabel('# Qubits')
ax1.set_zlabel('Processing time')
ax1.view_init(16,-75)
fig.colorbar(img, shrink=0.5, label='Stride', ticks=np.unique(st))

ax2 = fig.add_subplot(132, projection = '3d')
img2 = ax2.scatter(s,q,t, c=k, cmap='viridis', vmin=2, vmax=4)
ax2.set_xlabel('Image Size')
ax2.set_ylabel('# Qubits')
ax2.set_zlabel('Processing time')
ax2.view_init(16,-75)
fig.colorbar(img2, shrink=0.5, label='Kernel Size', ticks=np.unique(k))

ax3 = fig.add_subplot(133, projection = '3d')
img3 = ax3.scatter(s,q,t, c=f, cmap='viridis', vmin=4, vmax=16)
ax3.set_xlabel('Image Size')
ax3.set_ylabel('# Qubits')
ax3.set_zlabel('Processing time')
ax3.view_init(16,-75)
fig.colorbar(img3, shrink=0.5, label='# Features Map', ticks=np.unique(f))

fig.tight_layout()
plt.show()



markers = ['o', 'd', '+', 'x']

fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (10,5))
df2 = df[df.qubits == 4]
df2 = df2[df2.kernelsize == 2]
df2 = df2[df2.filters == 4]
q  = df2.qubits
f  = df2.filters
k  = df2.kernelsize.values.astype(np.int32)
st = df2.stride.values.astype(np.int32)
s  = df2.imgsize.values.astype(np.int32)
t  = df2.time

img1 = ax[0].scatter(s, t, c=st)
ax[0].set_xlabel('Image Size')
ax[0].set_ylabel('Processing Time [s]')
ax[0].set_title('Processing Time for circuit of 4 Qubits and Kernel Size 2$x$2')
ax[0].grid()
fig.colorbar(img1, shrink=0.8, label='Stride', ticks=np.unique(st))

df2 = df[df.qubits == 16]
df2 = df2[df2.kernelsize == 2]
df2 = df2[df2.filters == 4]
q  = df2.qubits
f  = df2.filters
k  = df2.kernelsize.values.astype(np.int32)
st = df2.stride.values.astype(np.int32)
s  = df2.imgsize.values.astype(np.int32)
t  = df2.time

img2 = ax[1].scatter(s, t, c=st)
ax[1].set_xlabel('Image Size')
ax[1].set_ylabel('Processing Time [s]')
ax[1].set_title('Processing Time for circuit of 16 Qubits and Kernel Size 2$x$2')
ax[1].grid()
fig.colorbar(img2, shrink=0.8, label='Stride', ticks=np.unique(st))

fig.tight_layout()
plt.show()