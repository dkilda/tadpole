#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as spla

import tadpole.array.backends.util    as util
import tadpole.array.backends.backend as backend 




# --- Numpy backend --------------------------------------------------------- #

class NumpyBackend(backend.Backend):

   # --- Core methods --- #

   def name(self):

       return "numpy"
       

   def copy(self, array, **opts):

       return np.copy(array, **opts)
       

   # --- Data type methods --- #

   def astype(self, array, **opts):

       dtype = self.get_dtype(opts.pop("dtype", None))

       try:
          return array.astype(dtype, **opts)
       except AttributeError:
          return self.asarray(array).astype(dtype)


   def dtype(self, array):

       try:   
          return array.dtype
       except AttributeError:
          return np.result_type(array)


   def iscomplex(self, array):

       return self.dtype(array) in self.complex_dtypes()


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


   def zeros(self, shape, **opts):

       dtype = self.get_dtype(opts.pop("dtype", None))
       return np.zeros(shape, dtype=dtype, **opts)
       

   def ones(self, shape, **opts):

       dtype = self.get_dtype(opts.pop("dtype", None))
       return np.ones(shape, dtype=dtype, **opts)       


   def unit(self, shape, idx, **opts):

       return util.unit(self, shape, idx, **opts)
       

   def eye(self, N, M=None, **opts):

       dtype = self.get_dtype(opts.pop("dtype", None))
       return np.eye(N, M=M, dtype=dtype, **opts)

   
   def _rand_helper(self, fun, **opts):

       dtype = self.get_dtype(opts.pop("dtype", None))
       seed  = opts.pop("seed", None)

       if seed is not None:
          np.random.seed(seed)
   
       def array(dtype):
           return self.astype(fun(**opts), dtype=dtype)

       if dtype in self.complex_dtypes():
          return array(dtype) + 1j*array(dtype) 
        
       return array(dtype) 


   def rand(self, shape, **opts):

       def fun(**kwargs): 
           return np.random.rand(*shape, **kwargs)

       return self._rand_helper(fun, **opts)
       

   def randn(self, shape, **opts):

       def fun(**kwargs): 
           return np.random.randn(*shape, **kwargs)

       return self._rand_helper(fun, **opts)
       

   def randuniform(self, shape, boundaries, **opts):

       def fun(**kwargs): 
           return np.random.uniform(
                                    boundaries[0], 
                                    boundaries[1], 
                                    tuple(shape), 
                                    **kwargs
                                   )

       return self._rand_helper(fun, **opts)
       

   # --- Array shape methods --- #

   def size(self, array):

       return array.size


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

       
   def sumover(self, array, axis=None, dtype=None, **opts):

       return np.sum(array, axis, dtype, **opts) 


   def cumsum(self, array, axis=None, dtype=None, **opts):

       return np.cumsum(array, axis, dtype, **opts) 


   def broadcast_to(self, array, shape):

       return np.broadcast_to(array, shape)
       

   # --- Array value methods --- #

   def item(self, array, *idx):

       if len(idx) > 0:
          return array[idx]

       if self.size(array) > 1:
          raise ValueError(
             f"NumpyBackend.item(): can only convert array of "
             f"size 1 to scalar, but array size is {self.size(array)}"
          )

       return array.item()


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
      
 
   def put(self, array, idxs, vals, accumulate=False):

       out = array.copy()

       if accumulate:
          np.add.at(out, idxs, vals)
          return out

       out[idxs] = vals 
       return out


   def where(self, condition, x, y):
       
       return np.where(condition, x, y)


   def argsort(self, array, axis=None, **opts):

       if axis is None:
          axis = -1

       return np.argsort(array, axis=axis, **opts)
      

   def diag(self, array, **opts):

       return np.diag(array, **opts) 


   # --- Logical operations --- #

   def allclose(self, x, y, **opts):

       return np.allclose(x, y, **opts)


   def isclose(self, x, y, **opts):

       return np.isclose(x, y, **opts)


   def allequal(self, x, y):

       return np.array_equal(x, y)


   def isequal(self, x, y):

       return x == y


   def notequal(self, x, y):

       return x != y


   def greater(self, x, y):
       
       return x > y


   def less(self, x, y):
       
       return x < y


   def greater_equal(self, x, y):
       
       return x >= y


   def less_equal(self, x, y):
       
       return x <= y


   def logical_and(self, x, y):

       return np.logical_and(x, y)


   def logical_or(self, x, y):

       return np.logical_or(x, y)


   # --- Standard math --- #

   def conj(self, array, **opts):

       return np.conj(array, **opts) 


   def real(self, array):

       return np.real(array)


   def imag(self, array):

       return np.imag(array)
       

   def sqrt(self, array):

       return np.sqrt(array) 


   def log(self, array): 

       return np.log(array)


   def exp(self, array): 

       return np.exp(array)


   def floor(self, array):

       return np.floor(array)


   def neg(self, array):

       return np.negative(array)


   def sin(self, array):
       
       return np.sin(array)


   def cos(self, array):

       return np.cos(array)


   def tan(self, array):

       return np.tan(array)


   def arcsin(self, array):
       
       return np.arcsin(array)


   def arccos(self, array):

       return np.arccos(array)


   def arctan(self, array):

       return np.arctan(array)


   def sinh(self, array):

       return np.sinh(array)


   def cosh(self, array):

       return np.cosh(array)


   def tanh(self, array):

       return np.tanh(array)


   def arcsinh(self, array):

       return np.arcsinh(array)


   def arccosh(self, array):

       return np.arccosh(array)


   def arctanh(self, array):

       return np.arctanh(array)


   # --- Binary elementwise algebra --- #

   def add(self, x, y):

       return x + y
        

   def sub(self, x, y):

       return x - y        


   def mul(self, x, y):
       
       return x * y


   def div(self, x, y):

       return x / y


   def mod(self, x, y):

       return x % y


   def floordiv(self, x, y):

       return x // y


   def power(self, x, y):

       return np.power(x, y)
       

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

       return util.eig(self, lambda v: np.linalg.eigh(v), x)
       

   def eigh(self, x):

       return util.eigh(self, lambda v: np.linalg.eigh(v), x)


   # --- Linear algebra: matrix exponential --- #

   def expm(self, x):

       return spla.expm(x)       


   # --- Linear algebra: misc methods --- #

   def norm(self, x, axis=None, order=None, **opts):

       return np.linalg.norm(x, order, axis, **opts)


   def htranspose(self, x, axes):

       return self.transpose(self.conj(x), axes)




