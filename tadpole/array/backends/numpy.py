#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as spla

import tadpole.array.backends.common  as common
import tadpole.array.backends.backend as backend 




# --- Numpy backend --------------------------------------------------------- #

class NumpyBackend(backend.Backend):

   # --- Core methods --- #

   def name(self):

       return "numpy"
       

   def copy(self, array, **opts):

       return np.copy(array, **opts)
       

   # --- Data type methods --- #

   def dtype(self, array):

       return array.dtype

  
   def get_dtype(self, dtype):

       if  dtype is None:
           return np.float64

       return np.dtype(dtype)

  
   def complex_dtypes(self):

       dtypes = (np.complex64, np.complex128, np.complex256)

       return tuple(map(np.dtype, dtypes))
       

   # --- Array creation methods --- #

   def asarray(self, array, **opts):

       return np.asarray(array, **opts)


   def astype(self, array, **opts):

       dtype = self.get_dtype(opts.pop("dtype", None))
       return array.astype(dtype, **opts)
       

   def zeros(self, shape, **opts):

       dtype = self.get_dtype(opts.pop("dtype", None))
       return np.zeros(shape, dtype=dtype, **opts)
       

   def ones(self, shape, **opts):

       dtype = self.get_dtype(opts.pop("dtype", None))
       return np.ones(shape, dtype=dtype, **opts)       


   def unit(self, shape, idx, **opts):

       return common.unit(self, shape, idx, **opts)
       

   def eye(self, N, M=None, **opts):

       dtype = self.get_dtype(opts.pop("dtype", None))
       return np.eye(N, M=M, dtype=dtype, **opts)
       

   def diag(self, array, **opts):

       return np.diag(array, **opts) 
  
 
   def _rand_helper(self, fun, **opts):

       dtype = self.get_dtype(opts.pop("dtype", None))
       seed  = opts.pop("seed", None)

       if dtype in self.complex_dtypes():

          return fun(**opts).astype(dtype) + 1j*fun(**opts).astype(dtype)

       return fun(**opts).astype(dtype)


   def rand(self, shape, **opts):

       def fun(**kwargs): 
           return np.random.rand(*shape, **kwargs)

       return self._rand_helper(fun, **opts)
       

   def randn(self, shape, **opts):

       def fun(**kwargs): 
           return np.random.randn(*shape, **kwargs)

       return self._rand_helper(fun, **opts)
       

   def randuniform(self, shapes, boundaries, **opts):

       def fun(**kwargs): 
           return np.random.uniform(
                                    boundaries[0], 
                                    boundaries[1], 
                                    tuple(shape), 
                                    **kwargs
                                   )

       return self._rand_helper(fun, **opts)
       

   # --- Array shape methods --- #

   def ndim(self, array):

       return array.ndim
       

   def shape(self, array):

       return array.shape


   def reshape(self, array, shape, **opts):

       shape = np.asarray(shape).astype(np.int32)
       return np.reshape(array, shape, **opts) 
       

   def transpose(self, array, axes):

       return np.transpose(array, axes)
       

   def moveaxis(self, array, source, destination):

       return np.moveaxis(array, source, destination)


   def squeeze(self, array, axis=None):

       return np.squeeze(array, axis)
       

   def unsqueeze(self, array, axis):

       return np.expand_dims(array, axis)
       

   # --- Array value methods --- #

   def all(self, array, axis=None, **opts):

       return np.all(array, axis=axis, **opts)
       

   def any(self, array, axis=None, **opts):

       return np.any(array, axis=axis, **opts)


   def max(self, array, axis=None, **opts):

       return np.amax(array, axis, **opts)


   def min(self, array, axis=None, **opts):

       return np.amin(array, axis, **opts)
       

   def sign(self, array, **opts):

       return np.sign(array, **opts)


   def abs(self, array, **opts):

       return np.abs(array, **opts)


   def flip(self, array, axis=None):

       return np.flip(array, axis)


   def clip(self, array, minval, maxval, **opts):

       return np.clip(array, minval, maxval, **opts)


   def count_nonzero(self, array, axis=None, **opts):

       return np.count_nonzero(array, axis, **opts)
       

   # --- Simple math operations --- #

   def conj(self, array, **opts):

       return np.conj(array, **opts) 
       

   def sqrt(self, array, **opts):

       return np.sqrt(array, **opts) 
       

   def sumover(self, array, axis=None, dtype=None, **opts):

       return np.sum(array, axis, dtype, **opts) 


   def cumsum(self, array, axis=None, dtype=None, **opts):

       return np.cumsum(array, axis, dtype, **opts) 


   # --- Binary elementwise algebra --- #

   def add(self, x, y):

       return x + y
        

   def sub(self, x, y):

       return x - y        


   def mul(self, x, y):
       
       return x * y


   def div(self, x, y):

       return x / y
       

   # --- Linear algebra: multiplication methods --- #


   def einsum(self, equation, *xs, optimize=True):

       return np.einsum(equation, *xs, optimize=optimize)
       

   def dot(self, x, y):

       return np.dot(x, y)
       

   def kron(self, x, y):

       return np.kron(x, y)       


   # --- Linear algebra: decomposition methods --- #

   def svd(self, x):

       try:
          return np.linalg.svd(x, full_matrices=False)

       except (np.linalg.LinAlgError, ValueError) as err:

          if  isinstance(err, np.linalg.LinAlgError) or 'converge' in str(err):
              import warnings
              warnings.warn("NumPy SVD failed, trying " 
                            "again with alternative driver")

          return spla.svd(x, full_matrices=False, lapack_driver='gesvd')


   def qr(self, x):

       return np.linalg.qr(x, mode='reduced')


   def eig(self, x):

       return common.eig(self, lambda v: np.linalg.eigh(v), x)
       

   def eigh(self, x):

       return common.eigh(self, lambda v: np.linalg.eigh(v), x)


   # --- Linear algebra: matrix exponential --- #

   def expm(self, x):

       return spla.expm(x)       


   # --- Linear algebra: misc methods --- #

   def htranspose(self, x, axes):

       return self.transpose(self.conj(x), axes)
       

   def norm(self, x, order=None, axis=None, **opts):

       return np.linalg.norm(array, order, axis, **opts)




