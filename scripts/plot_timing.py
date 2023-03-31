from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


df = pd.read_csv('results.csv')
df = df[df.stride==3]

fig= plt.figure(figsize = (10,10))
ax = fig.add_subplot(111, projection = '3d')

q  = df.qubits
f  = df.filters
k  = df.kernelsize
st = df.stride
s  = df.imgsize
t  = df.time


img = ax.scatter(k, f, q, c=t, cmap='jet', s=s)

ax.set_zlabel('# Qubits')
ax.set_ylabel('# Filters')
ax.set_xlabel('Kernel Size')

fig.colorbar(img, shrink=0.5, label='Processing Time (s)')
plt.show()