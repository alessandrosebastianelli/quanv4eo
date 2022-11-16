from joblib import Parallel, delayed
from tqdm.auto import tqdm
import numpy as np

def unwrap_self(image, c, j, i, qcircuit, filters, ksize):
    '''
        Joblib or Multiprocessing is based on pickling to pass functions around to achieve parallelization. 
        In order to pickle the object, this object must capable of being referred to in the global context for
        the unpickle to be able to access it. The function we want to parallel above is not in global context,
        therefore, causing an error. Therefore, one solution I found
        (http://qingkaikong.blogspot.com/2016/12/python-parallel-method-in-class.html) 
        is to create a function outside the class to unpack the self from the arguments and calls the function again.
    '''
    return QConv3D.qConv3D2(image, c, j, i, qcircuit, filters, ksize)

class QConv3D:
    '''
        Quantum Convolution 2D
    '''

    def __init__(self, circuits, filters, kernel_size, stride, parallelize=0, reshape=True):
        '''
            Quantum Convolution 2D layer:
            
            - circuit: quantum circuit
            - filters: number of filters for the convolution, must not exceed the number of qubits
            - kernel_size: size of the kernel for the convolutio, must not exceed the square root of number of qubits
            - stride: value for stride in convolution
            - parallelize: if == 0 no parallelization, otherwise parallize with workers=parallelize
            - rehspae: if True will merge the features maps for each chanell
        '''
        self.circuits    = circuits
        self.filters     = filters
        self.kernel_size = kernel_size
        self.stride      = stride
        self.parallelize = parallelize
        self.reshape     = reshape

    def apply(self, image, verbose=False):
        '''
            Apply the Quantum Convolution to the image

            Input: 
                - image: a 3D channel-last matrix in R^(w,h,c), w: width, h: height, c:channels
            Output:
                - quantum convolved image: a 3D chanell-last matrix in R^(wc,hc,f), wc: width, hc: height, f:channels
        '''
        # There are two versions, one parellelized an one not

        results = []
        for circuit in self.circuits:
            if self.parallelize == 0:
                results.append(self.__qConv3D(image, circuit, verbose, self.reshape))
            else:
                results.append(self.par_qConv3D(image, circuit, self.filters, self.kernel_size, self.stride, self.parallelize, verbose, self.reshape))
        
        results = np.moveaxis(results, 0, -1)
        s = np.shape(results)
        return np.reshape(results, (s[0], s[1], s[-2]*s[-1]))

    @staticmethod
    def par_qConv3D(image, qcircuit, filters, ksize, stride, njobs, verbose, reshape):
        '''
            ###########################################################################
            # !!! This method is ment to be private, use the .apply method insted !!! #
            ###########################################################################

            Parallelize the quantum convolution.

            Inputs:
                - image: a 3D channel-last matrix in R^(w,h,c), w: width, h: height, c:channels
                - c: channel index from paralallization
                - i: row index from parallelization
                - j: colum index from parallelization
                - qcircuit: quantum circuit
                - filters: number of filters for the convolution, must not exceed the number of qubits
                - kernel_size: size of the kernel for the convolutio, must not exceed the square root of number of qubits
                - stride: value for stride in convolution
                - njobs: number of workers for parallelization
                - verbose: if True print tqdm progress bar
            Output:
                - res: quantum convolved image
        '''

        # Calculates the image shape after the convolution
        h, w, ch = image.shape
        h_out = (h-ksize) // stride + 1
        w_out = (w-ksize) // stride + 1 
        # Embedding x and y spatial loops and the spectral loop into Joblib
        res = Parallel(n_jobs=njobs)( 
            delayed(unwrap_self)(image, c, j, i, qcircuit, filters, ksize) 
            for j in tqdm(range(0, h-ksize, stride), disable=not(verbose), leave=False, colour='black')
            for i in range(0, w-ksize, stride)
            for c in range(ch)
        )

        # Joblib returns a 1-D array, the following functions are used to reshape the convolution output into the correct shape
        res = np.array(res).flatten()
        try:
            if reshape: res = res.reshape((h_out, w_out, ch * filters))
        except:
            if reshape: res = res.reshape((h_out-1, w_out-1, ch * filters))

        return res
    
    @staticmethod
    def qConv3D2(image, c, j, i, qcircuit, filters, ksize):
        '''
            ###########################################################################
            # !!! This method is ment to be private, use the .apply method insted !!! #
            ###########################################################################

            Applies the quantum circuit to a small portion of the input image.

            Inputs:
                - image: a 3D channel-last matrix in R^(w,h,c), w: width, h: height, c:channels
                - c: channel index from paralallization
                - i: row index from parallelization
                - j: colum index from parallelization
                - qcircuit: quantum circuit
                - filters: number of filters for the convolution, must not exceed the number of qubits
                - kernel_size: size of the kernel for the convolutio, must not exceed the square root of number of qubits
            Output:
                - q_results: 1D quantum output vector
        '''
       #out = np.zeros((filters))
        p = image[j:j+ksize, i:i+ksize, c]
        q_results = qcircuit(p.reshape(-1))
        
        #for k in range(filters):
        #    out[k] = q_results[k]
            
        return q_results

    def __qConv3D(self, image, qcircuit, verbose, reshape):

        '''
            Non parallelized quantum convolution.

            Inputs:
                - image: a 3D channel-last matrix in R^(w,h,c), w: width, h: height, c:channels
                - verbose: if True print tqdm progress bar
            Output:
                - q_results: 1D quantum output vector
        '''
        # Calculates the image shape after the convolution
        h, w, ch = image.shape
        h_out = (h-self.kernel_size) // self.stride +1
        w_out = (w-self.kernel_size) // self.stride +1

        out = np.zeros((h_out, w_out, ch, self.filters))
        
        ctx = 0
        cty = 0
        # Spectral Loop
        for c in tqdm(range(ch), desc='Channel', disable=not(verbose), colour='black'):
            # Spatial Loops                                                
            for j in tqdm(range(0, h-self.kernel_size, self.stride), desc='Column', leave=False, disable=not(verbose), colour='black'):
                for i in tqdm(range(0, w-self.kernel_size, self.stride), desc='Row', leave=False, disable=not(verbose), colour='black'):            
                    # Process a kernel_size*kernel_size region of the images
                    # with the quantum circuit stride*stride
                    p = image[j:j+self.kernel_size, i:i+self.kernel_size, c]

                    q_results = qcircuit(p.reshape(-1))

                    for k in range(self.filters):
                        out[cty, ctx, c, k] = q_results[k]

                    ctx+=1
                ctx = 0
                cty += 1
            
            ctx = 0
            cty = 0

        if reshape: out = out.reshape((h_out, w_out, ch * self.filters))

        return out
