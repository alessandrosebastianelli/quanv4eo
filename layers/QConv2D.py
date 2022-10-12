from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np

class qConv2D:

    def __init__(self, circuit, filters, kernel_size, stride, parallelize=0):
        self.circuit     = circuit
        self.filters     = filters
        self.kernel_size = kernel_size
        self.stride      = stride
        self.parallelize = parallelize

    def apply(self, image):

        if self.parallelize == 0:
            return self.__qConv2D(image)
        else:
            return self.__par_qConv2D(image)
        pass

    def __par_qConv2D(self, image):
        h, w, ch = image.shape
        h_out = (h-self.kernel_size) // self.stride + 1
        w_out = (w-self.kernel_size) // self.stride + 1 
        
        res = Parallel(n_jobs=self.parallelize)( 
        delayed(qConv2D2)(image, c, j, i) for j in range(0, h-self.kernel_size, self.stride) for i in range(0, w-self.kernel_size, self.stride) for c in range(ch))

        return np.array(res).flatten()

    def __qConv2D2(self, image, c, j, i):
        out = np.zeros((self.filters))
        p = image[j:j+self.kernel_size, i:i+self.kernel_size, c]
        q_results = self.circuit(p.reshape(-1))

        for k in range(self.filters):
            out[k] = q_results[k]
            
        return out

    def __qConv2D(self, image):
        '''
            Convolves an image with many applications of the same
            quantum circuit
        '''
        h, w, ch = image.shape
        h_out = (h-self.kernel_size) // self.stride +1
        w_out = (w-self.kernel_size) // self.stride +1

        out = np.zeros((h_out, w_out, self.filters, ch))
        
        ctx = 0
        cty = 0
        # Spectral Loop
        for c in range(ch):
            # Spatial Loops
            for j in range(0, h-self.kernel_size, self.stride):
                for i in range(0, w-self.kernel_size, self.stride):            
                    # Process a kernel_size*kernel_size region of the images
                    # with the quantum circuit stride*stride
                    p = image[j:j+self.kernel_size, i:i+self.kernel_size, c]

                    q_results = self.circuit(p.reshape(-1))

                    for k in range(filters):
                        out[cty, ctx, k, c] = q_results[k]

                    ctx+=1
                ctx = 0
                cty += 1
            
            ctx = 0
            cty = 0

        out = np.mean(out, -1, keepdims = False)

        return out