from tqdm.auto import tqdm
import numpy as np
import glob
import os




root     = '/Volumes/PortableSSD/datasets/quantum-processed-dataset'
shape    = (61,61,8)


datasets = glob.glob(os.path.join(root, '*'))
paths = []
datasets_name = []

for dataset in datasets:
    p = glob.glob(os.path.join(dataset, '*', '*'))
    p.sort()
    paths.append(p)
    datasets_name.append(dataset.split(os.sep)[-1])


print(datasets_name)

for ii in tqdm(range(len(paths[0]))):
    for jj, n in enumerate(datasets_name):
        
        if n!='processed':
            if jj == 0:
                img = np.load(paths[jj][ii], allow_pickle = True)
                #print(img.shape, jj)
            else:
                imgjj = np.load(paths[jj][ii], allow_pickle = True)
                #print(imgjj.shape, jj)
                img = np.concatenate((img, imgjj), axis = -1)
        
    svpath = paths[0][ii].replace(datasets_name[0], 'processed')
    #print(os.path.join(os.sep, *svpath.split(os.sep)[:-1]))
    os.makedirs(os.path.join(os.sep, *svpath.split(os.sep)[:-1]), exist_ok=True)
    np.save(svpath, img)
    #print(img.shape)
   # break







#print(paths[0][0])
#print(paths[1][0])
#print(paths[0][1000])
#print(paths[1][1000])


        



