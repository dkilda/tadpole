#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np

import tadpole.util     as util
import tadpole.autodiff as ad



# --- ArrayLike interface --------------------------------------------------- #

class ArrayLike(abc.ABC):

   @abc.abstractmethod
   def new(self, data):
       pass

   @abc.abstractmethod
   def __or__(self, other):
       pass
   



# --- Single Array (supports unary operations) ------------------------------ #

class OneArray(ArrayLike):

   # --- Construction --- #

   def __init__(self, backend, data):

       if not isinstance(backend, backends.backend.Backend):
          raise ValueError(
             f"OneArray: backend must be an instance "
             f"of Backend, but it is {backend}"
          ) 

       self._backend = backend
       self._data    = data


   # --- Arraylike methods --- #

   def new(self, data):

       return self.__class__(self._backend, data)


   def __or__(self, other):

       backend = self._backend # TODO work out the common backend

       if isinstance(other, OneArray):
          return TwoArray(backend, self._data, other._data)

       return NArray(backend, self._data, *other._datas)


   @property
   def _datas(self):

       return (self._data, )


   # --- Core methods --- #

   def copy(self, **opts):

       data = self._backend.copy(self._data, **opts)

       return self.new(data) 


   # --- Data type methods --- #

   def astype(self, **opts):

       data = self._data.astype(**opts)

       return self.new(data) 


   def dtype(self):

       return str(self._backend.dtype(self._data))


   def iscomplex(self):

       return self._backend.iscomplex(self._data)


   # --- Shape methods --- #

   def size(self):

       return self._backend.size(self._data)


   def ndim(self):

       return self._backend.ndim(self._data)


   def shape(self):

       return self._backend.shape(self._data)


   def reshape(self, shape, **opts):

       data = self._backend.reshape(self._data, shape, **opts)

       return self.new(data) 


   def transpose(self, axes):

       data = self._backend.transpose(self._data, axes)

       return self.new(data) 


   def moveaxis(self, source, destination):

       data = self._backend.moveaxis(self._data, source, destination)

       return self.new(data) 


   def squeeze(self, axis=None):

       data = self._backend.squeeze(self._data, axis)

       return self.new(data) 


   def unsqueeze(self, axis):

       data = self._backend.unsqueeze(self._data, axis)

       return self.new(data) 


   def sumover(self, axis=None, dtype=None, **opts):

       data = self._backend.sumover(self._data, axis, dtype, **opts) 

       return self.new(data) 


   def cumsum(self, axis=None, dtype=None, **opts):

       data = self._backend.sumover(self._data, axis, dtype, **opts) 

       return self.new(data) 


   # --- Value methods --- #

   def item(self, *idx):

       return self._backend.item(self._data, *idx)


   def all(self, axis=None, **opts):

       data = self._backend.all(self._data, axis, **opts)

       return self.new(data) 


   def any(self, axis=None, **opts):

       data = self._backend.any(self._data, axis, **opts)

       return self.new(data) 


   def max(self, axis=None, **opts):

       return self._backend.max(self._data, axis, **opts)


   def min(self, axis=None, **opts):

       return self._backend.min(self._data, axis, **opts)


   def sign(self, **opts):

       data = self._backend.sign(self._data, **opts)

       return self.new(data) 


   def abs(self, **opts):

       data = self._backend.abs(self._data, **opts)

       return self.new(data) 


   def flip(self, axis=None):

       data = self._backend.flip(self._data, axis)

       return self.new(data) 


   def clip(self, minval, maxval, **opts):

       data = self._backend.clip(self._data, minval, maxval, **opts):

       return self.new(data) 


   def count_nonzero(self, axis=None, **opts):

       data = self._backend.count_nonzero(self._data, axis, **opts)

       return self.new(data) 


   def put(self, idxs, vals, accumulate=False):

       data = self._backend.put(
                 self._data, idxs, vals, accumulate=accumulate
              )

       return self.new(data) 


   def argsort(self, axis=-1, **opts):

       data = self._backend.argsort(self._data, axis, **opts)

       return self.new(data)         


   # --- Standard math --- #

   def conj(self):

       data = self._backend.conj(self._data)

       return self.new(data) 


   def real(self):

       data = self._backend.real(self._data)

       return self.new(data) 


   def imag(self):

       data = self._backend.imag(self._data)

       return self.new(data) 
       

   def sqrt(self):

       data = self._backend.sqrt(self._data)

       return self.new(data) 


   def log(self): 

       data = self._backend.log(self._data)

       return self.new(data) 


   def exp(self): 

       data = self._backend.exp(self._data)

       return self.new(data) 


   def neg(self):

       data = self._backend.neg(self._data)

       return self.new(data) 


   def sin(self):

       data = self._backend.sin(self._data)
       
       return self.new(data) 


   def cos(self):

       data = self._backend.cos(self._data)

       return self.new(data) 


   def tan(self):

       data = self._backend.tan(self._data)

       return self.new(data) 


   def arcsin(self):

       data = self._backend.arcsin(self._data)
       
       return self.new(data) 


   def arccos(self):

       data = self._backend.arccos(self._data)

       return self.new(data) 


   def arctan(self):

       data = self._backend.arctan(self._data)

       return self.new(data) 


   def sinh(self):

       data = self._backend.sinh(self._data)

       return self.new(data) 


   def cosh(self):

       data = self._backend.cosh(self._data)

       return self.new(data) 


   def tanh(self):

       data = self._backend.tanh(self._data)

       return self.new(data) 


   def arcsinh(self):

       data = self._backend.arcsinh(self._data)

       return self.new(data) 


   def arccosh(self):

       data = self._backend.arccosh(self._data)

       return self.new(data) 


   def arctanh(self):

       data = self._backend.arctanh(self._data)

       return self.new(data) 


   # --- Linear algebra: decompositions --- #

   def svd(self):

       U, S, VH = self._backend.svd(self._data)

       return self.new(U), self.new(S), self.new(VH)


   def qr(self):

       Q, R = self._backend.qr(self._data)

       return self.new(Q), self.new(R)


   def eig(self):

       U, S, VH = self._backend.eig(self._data)

       return self.new(U), self.new(S), self.new(VH) 


   def eigh(self):

       U, S, VH = self._backend.eigh(self._data)

       return self.new(U), self.new(S), self.new(VH) 


   # --- Linear algebra: matrix exponential --- #

   def expm(self):

       data = self._backend.expm(self._data)

       return self.new(data)


   # --- Linear algebra: norm --- #

   def norm(self, order=None, axis=None, **opts):

       data = self._backend.norm(self._data, order, axis, **opts)

       return self.new(data)       




# --- Double Array (supports binary operations) ----------------------------- #

class TwoArray(ArrayLike):

   # --- Construction --- #

   def __init__(self, backend, dataA, dataB):

       self._backend = backend
       self._datas   = (dataA, dataB)


   # --- Arraylike methods --- #

   def new(self, data):

       return OneArray(self._backend, data)


   def __or__(self, other):

       backend = self._backend # TODO work out the common backend

       return NArray(backend, *self._datas, *other._datas)


   # --- Logical operations --- #

   def allclose(self, **opts):

       return self._backend.allclose(*self._datas, **opts)  


   def allequal(self):

       return self._backend.allequal(*self._datas) 


   def isclose(self, **opts):

       data = self._backend.isclose(*self._datas, **opts)  

       return self.new(data)
 

   def isequal(self):

       data = self._backend.isequal(*self._datas) 
 
       return self.new(data)


   def notequal(self):

       data = self._backend.notequal(*self._datas)

       return self.new(data)


   def logical_and(self):

       data = self._backend.logical_and(*self._datas)

       return self.new(data)


   def logical_or(self):

       data = self._backend.logical_or(*self._datas)

       return self.new(data)


   # --- Elementwise binary algebra --- #

   def add(self):

       data = self._backend.add(*self._datas)

       return self.new(data)

       
   def sub(self):

       data = self._backend.sub(*self._datas)

       return self.new(data)      


   def mul(self):
       
       data = self._backend.mul(*self._datas)

       return self.new(data)


   def div(self):

       data = self._backend.div(*self._datas)

       return self.new(data)


   def power(self):

       data = self._backend.power(*self._datas)

       return self.new(data)


   # --- Linear algebra: products --- #

   def dot(self):

       data = self._backend.dot(*self._datas)

       return self.new(data)
       

   def kron(self):

       data = self._backend.kron(*self._datas)

       return self.new(data)






# --- NTuple Array (supports nary operations) ------------------------------- #

class NArray(ArrayLike):

   # --- Construction --- #

   def __init__(self, backend, *datas):

       self._backend = backend
       self._datas   = datas


   # --- Arraylike methods --- #

   def new(self, data):

       return OneArray(self._backend, data)


   def __or__(self, other):

       backend = self._backend # TODO work out the common backend

       return NArray(backend, *self._datas, *other._datas)


   # --- Value methods --- #

   def where(self): 

       data = self._backend.where(*self.datas)

       return self.new(data)


   # --- Linear algebra: products --- #

   def einsum(self, equation, optimize=True):

       data = self._backend.einsum(equation, *self._datas, optimize=optimize)

       return self.new(data)































































