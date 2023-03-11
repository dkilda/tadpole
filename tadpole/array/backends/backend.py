#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc




# --- Backend interface ----------------------------------------------------- #

class Backend(abc.ABC):

   # --- Core methods --- #

   @abc.abstractmethod
   def name(self):
       pass

   @abc.abstractmethod   
   def copy(self, array, **opts):
       pass


   # --- Data type methods --- #

   @abc.abstractmethod  
   def dtype(self, array):
       pass

   @abc.abstractmethod  
   def get_dtype(self, dtype):
       pass

   @abc.abstractmethod  
   def complex_dtypes(self):
       pass


   # --- Array creation methods --- #

   @abc.abstractmethod
   def asarray(self, array, **opts):
       pass

   @abc.abstractmethod
   def astype(self, array, **opts):
       pass

   @abc.abstractmethod
   def zeros(self, shape, **opts):
       pass

   @abc.abstractmethod
   def ones(self, shape, **opts):
       pass

   @abc.abstractmethod
   def unit(self, shape, idx, **opts):
       pass

   @abc.abstractmethod
   def eye(self, N, M=None, **opts):
       pass

   @abc.abstractmethod
   def diag(self, array, **opts):
       pass

   @abc.abstractmethod
   def rand(self, shape, **opts):
       pass

   @abc.abstractmethod
   def randn(self, shape, **opts):
       pass

   @abc.abstractmethod
   def randuniform(self, shapes, boundaries, **opts):
       pass


   # --- Array shape methods --- #

   @abc.abstractmethod
   def size(self, array):
       pass

   @abc.abstractmethod
   def ndim(self, array):
       pass

   @abc.abstractmethod
   def shape(self, array):
       pass

   @abc.abstractmethod
   def reshape(self, array, shape, **opts):
       pass

   @abc.abstractmethod
   def transpose(self, array, axes):
       pass

   @abc.abstractmethod
   def moveaxis(self, array, source, destination):
       pass

   @abc.abstractmethod
   def squeeze(self, array, axis=None):
       pass

   @abc.abstractmethod
   def unsqueeze(self, array, axis):
       pass


   # --- Array value methods --- #

   @abc.abstractmethod
   def item(self, array, *idx):
       pass

   @abc.abstractmethod
   def all(self, array, axis=None, **opts):
       pass

   @abc.abstractmethod
   def any(self, array, axis=None, **opts):
       pass

   @abc.abstractmethod
   def max(self, array, axis=None, **opts):
       pass

   @abc.abstractmethod
   def min(self, array, axis=None, **opts):
       pass

   @abc.abstractmethod
   def sign(self, array, **opts):
       pass

   @abc.abstractmethod
   def abs(self, array, **opts):
       pass

   @abc.abstractmethod
   def flip(self, array, axis=None):
       pass

   @abc.abstractmethod
   def clip(self, array, minval, maxval, **opts):
       pass

   @abc.abstractmethod
   def count_nonzero(self, array, axis=None, **opts):
       pass

   @abc.abstractmethod
   def put(self, array, idxs, vals, accumulate=False):
       pass

   @abc.abstractmethod
   def where(self, condition, x, y):
       pass

   @abc.abstractmethod
   def argsort(self, array, axis=-1, **opts):
       pass


   # --- Logical operations --- #

   @abc.abstractmethod
   def logical_and(self, x, y):
       pass

   @abc.abstractmethod
   def not_equal(self, x, y):
       pass


   # --- Simple math operations --- #

   @abc.abstractmethod
   def conj(self, array, **opts):
       pass

   @abc.abstractmethod
   def sqrt(self, array):
       pass

   @abc.abstractmethod
   def log(self, array):
       pass

   @abc.abstractmethod
   def exp(self, array):
       pass

   @abc.abstractmethod
   def neg(self, array):
       pass

   @abc.abstractmethod
   def sin(self, array):
       pass

   @abc.abstractmethod
   def cos(self, array):
       pass

   @abc.abstractmethod
   def tan(self, array):
       pass

   @abc.abstractmethod
   def arcsin(self, array):
       pass

   @abc.abstractmethod
   def arccos(self, array):
       pass

   @abc.abstractmethod
   def arctan(self, array):
       pass

   @abc.abstractmethod
   def sinh(self, array):
       pass

   @abc.abstractmethod
   def cosh(self, array):
       pass

   @abc.abstractmethod
   def tanh(self, array):
       pass

   @abc.abstractmethod
   def arcsinh(self, array):
       pass

   @abc.abstractmethod
   def arccosh(self, array):
       pass

   @abc.abstractmethod
   def arctanh(self, array):
       pass

   @abc.abstractmethod
   def sumover(self, array, axis=None, dtype=None, **opts):
       pass

   @abc.abstractmethod
   def cumsum(self, array, axis=None, dtype=None, **opts):
       pass


   # --- Binary elementwise algebra --- #

   @abc.abstractmethod
   def add(self, x, y):
       pass 

   @abc.abstractmethod
   def sub(self, x, y):
       pass 

   @abc.abstractmethod
   def mul(self, x, y):
       pass

   @abc.abstractmethod
   def div(self, x, y):
       pass

   @abc.abstractmethod
   def power(self, x, y):
       pass


   # --- Linear algebra: multiplication methods --- #

   @abc.abstractmethod
   def einsum(self, equation, *xs, optimize=True):
       pass

   @abc.abstractmethod
   def dot(self, x, y):
       pass

   @abc.abstractmethod
   def kron(self, x, y):
       pass


   # --- Linear algebra: decomposition methods --- #

   @abc.abstractmethod
   def svd(self, x):
       pass

   @abc.abstractmethod
   def qr(self, x):
       pass

   @abc.abstractmethod
   def eig(self, x):
       pass

   @abc.abstractmethod
   def eigh(self, x):
       pass


   # --- Linear algebra: matrix exponential --- #

   @abc.abstractmethod
   def expm(self, x):
       pass


   # --- Linear algebra: misc methods --- #

   @abc.abstractmethod
   def htranspose(self, x, axes):
       pass

   @abc.abstractmethod
   def norm(self, x, order=None, axis=None, **opts):
       pass




