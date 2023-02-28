#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import tadpole.array.backends.common  as common
import tadpole.array.backends.backend as backend 

torch = None



# --- Torch backend --------------------------------------------------------- #

class TorchBackend(backend.Backend):

   # --- Pseudo-constructor --- #

   def __init__(self):

       try:
           import torch as _torch
       except ImportError:
           raise ImportError("Torch is not installed. Please "
                             "install Torch or use a different backend.")

       torch = _torch


   # --- Core methods --- #

   def name(self):

       return "torch"
       

   def copy(self, array, **opts):

       return torch.detach().clone(array, **opts)       


   # --- Data type methods --- #

   def dtype(self, array):

       return array.dtype


   def _get_dtype(self, dtype):

       return {
           "float32":    torch.float32,
           "float64":    torch.float64,
           "complex64":  torch.complex64,
           "complex128": torch.complex128,
           "uint8":      torch.uint8,
           "int8":       torch.int8,
           "int16":      torch.int16,
           "int32":      torch.int32,
           "int64":      torch.int64,
           "float":      torch.float32,
           "double":     torch.float64,
           "cfloat":     torch.complex64,
           "cdouble":    torch.complex128,
           "short":      torch.int16,
           "int":        torch.int32,
           "long":       torch.int64,
           "bool":       torch.bool,
          }[dtype]
       

   def get_dtype(self, dtype):

       if  dtype is None:
           return torch.float64

       if  isinstance(dtype, torch.dtype):
           return dtype

       if  isinstance(dtype, str):
           return self._get_dtype(dtype)

       return self._get_dtype(np.dtype(dtype).name)
       

   def complex_dtypes(self):

       return (torch.complex64, torch.complex128)
       

   # --- Array creation methods --- #

   def asarray(self, array, **opts):

       return torch.as_tensor(array, **opts)
       

   def astype(self, array, **opts):

       dtype = self.get_dtype(opts.pop("dtype", None))
       return array.type(dtype, **opts)
       

   def zeros(self, shape, **opts):

       dtype = self.get_dtype(opts.pop("dtype", None))
       return torch.zeros(shape, dtype=dtype, **opts)
       

   def ones(self, shape, **opts):

       dtype = self.get_dtype(opts.pop("dtype", None))
       return torch.ones(shape, dtype=dtype, **opts)
       

   def unit(self, shape, idx, **opts):

       return common.unit(self, shape, idx, **opts)
       

   def eye(self, N, M=None, **opts):

       dtype = self.get_dtype(opts.pop("dtype", None))

       if  M is None:
           M = N

       return torch.eye(n=N, m=M, dtype=dtype, **opts)
       

   def diag(self, array, **opts):

       return torch.diag(array, **opts)


   def _rand_helper(self, fun, **opts):

       dtype = self.get_dtype(opts.pop("dtype", None))
       seed  = opts.pop("seed", None)

       if seed is not None:
          torch.manual_seed(seed)

       return fun(dtype=dtype, **opts)


   def rand(self, shape, **opts):

       def fun(**kwargs): 
           return torch.rand(*shape, **kwargs)

       return self._rand_helper(fun, **opts)
       

   def randn(self, shape, **opts):

       def fun(**kwargs): 
           return torch.randn(*shape, **kwargs)

       return self._rand_helper(fun, **opts)
       

   def randuniform(self, shapes, boundaries, **opts):

       def fun(**kwargs): 
           return torch.emply(tuple(shape), **kwargs).uniform_(*boundaries)

       return self._rand_helper(fun, **opts)
       

   # --- Array shape methods --- #

   def ndim(self, array):

       return array.dim()
       

   def shape(self, array):

       return array.size()       


   def reshape(self, array, shape, **opts):

       shape = tuple(np.array(shape).astype(int))
       return torch.reshape(array, shape, **opts) 


   def transpose(self, array, axes):
       
       return array.permute(axes)


   def moveaxis(self, array, source, destination):

       return torch.moveaxis(array, source, destination)
       

   def squeeze(self, array, axis=None):

       return torch.squeeze(array, axis)
       

   def unsqueeze(self, array, axis):

       return torch.unsqueeze(array, axis)
       

   # --- Array value methods --- #

   def all(self, array, axis=None, **opts):

       if axis is None:
          return torch.all(array)

       return torch.all(array, axis=axis, **opts)
       

   def any(self, array, axis=None, **opts):

       if axis is None:
          return torch.any(array)

       return torch.any(array, axis=axis, **opts)
       

   def max(self, array, axis=None, **opts):

       if axis is None:
          return torch.max(array)

       return torch.max(array, axis, **opts)
       

   def min(self, array, axis=None, **opts):

       if axis is None:
          return torch.min(array)

       return torch.min(array, axis, **opts)       


   def sign(self, array, **opts):

       return torch.sign(array, **opts)
       

   def abs(self, array, **opts):

       return torch.abs(array, **opts)
       

   def flip(self, array, axis=None):

       if axis is None:
          return torch.flip(array, tuple(range(array.dim())))

       return torch.flip(array, (axis,))
       

   def clip(self, array, minval, maxval, **opts):

       return torch.clip(array, minval, maxval, **opts)
       

   def count_nonzero(self, array, axis=None, **opts):

       return torch.count_nonzero(array, axis, **opts)


   def put(self, array, idxs, vals, accumulate=False):

       return array.index_put(idxs, vals, accumulate=accumulate)
       

   # --- Simple math operations --- #

   def conj(self, array, **opts):

       return torch.conj(array) 
       

   def sqrt(self, array, **opts):

       return torch.sqrt(array, **opts) 


   def neg(self, array):

       return torch.neg(array)


   def sin(self, array):
       
       return torch.sin(array)


   def cos(self, array):

       return torch.cos(array)
       

   def sumover(self, array, axis=None, dtype=None, **opts):

       if axis is None:
          return torch.sum(array, dtype)

       return torch.sum(array, axis, dtype, **opts) 
       

   def cumsum(self, array, axis=None, dtype=None, **opts):

       if axis is None:

          flat_array = torch.reshape(array, (-1))

          return torch.cumsum(flat_array, 0, dtype, **opts)

       return torch.cumsum(array, axis, dtype, **opts) 

       
   # --- Binary elementwise algebra --- #

   def add(self, x, y):

       return torch.add(x, y)
        

   def sub(self, x, y):
        
       return torch.sub(x, y)


   def mul(self, x, y):

       return torch.mul(x, y)
       

   def div(self, x, y):

       return torch.div(x, y)       


   # --- Linear algebra: multiplication methods --- #

   def einsum(self, equation, *xs, optimize=True):

       return torch.einsum(equation, *xs)
       

   def dot(self, x, y):

       return torch.dot(x, y)
       

   def kron(self, x, y):

       return torch.kron(x, y)
       

   # --- Linear algebra: decomposition methods --- #

   def svd(self, x):

       return torch.linalg.svd(x, full_matrices=False)
       

   def qr(self, x):

       return torch.linalg.qr(x, mode='reduced')
       

   def eig(self, x):

       return common.eig(self, lambda v: torch.linalg.eigh(v), x)
       

   def eigh(self, x):

       return common.eigh(self, lambda v: torch.linalg.eigh(v), x)
       

   # --- Linear algebra: matrix exponential --- #

   def expm(self, x):

       return torch.linalg.matrix_exp(x)
       

   # --- Linear algebra: misc methods --- #

   def htranspose(self, x, axes):

       return self.transpose(self.conj(x), axes)
       

   def norm(self, x, order=None, axis=None, **opts):

       if order is None:
          order = "fro"

       return torch.norm(x, p=order, dim=axis, **opts)
       


































































































