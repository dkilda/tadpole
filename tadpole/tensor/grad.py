#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np

import tadpole.util as util

import tadpole.tensor.backends as backends
import tadpole.tensor.core     as core

from tadpole.tensor.types import TensorLike, Pluggable




###############################################################################
###                                                                         ###
###  A general framework for tensor gradients                               ###
###                                                                         ###
###############################################################################


# --- Zero gradient --------------------------------------------------------- #

class ZeroGrad(TensorLike, Pluggable):

   # --- Construction --- #

   def __init__(self):

       self._backend = backends.get(None)


   # --- Plugging into function calls --- #

   def pluginto(self, funcall):

       return self.todense().pluginto(funcall)


   # --- Gradient accumulation --- #

   def addto(self, other):

       return other.addto(self)


   # --- Basic functionality --- #

   def copy(self):

       return self.__class__()


   def todense(self):

       return self.space().zeros()


   def withdata(self, data):

       return core.astensor(data, backend=self._backend)


   def space(self):

       return core.TensorSpace(
                              self._backend.name(), 
                              tuple(), 
                              self._backend.get_dtype(None)
                             )

   def item(self):

       return self.todense().item() 


   # --- Tensor properties --- #

   @property
   def dtype(self):
       return self.space().dtype

   @property 
   def size(self):
       return self.space().size

   @property 
   def ndim(self):
       return self.space().ndim

   @property 
   def shape(self):
       return self.space().shape


   # --- Comparisons --- #
       
   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       return bool(log)


   # --- Arithmetics and element access --- #

   def __getitem__(self, idx):

       return self.todense()[idx]


   def __neg__(self):

       return self.todense()


   def __add__(self, other): 

       return other


   def __sub__(self, other): 

       return -other 


   def __mul__(self, other):

       return self.todense()


   def __truediv__(self, other):

       return self.todense()


   def __pow__(self, other):

       return self.todense() 


   def __radd__(self, other): 

       return other  


   def __rsub__(self, other): 

       return other


   def __rmul__(self, other):

       return self.todense()


   def __rtruediv__(self, other):

       raise ValueError(
          "ZeroGrad.__rtruediv__: division by zero encountered!"
       )


   def __rpow__(self, other):

       if not isinstance(other, TensorLike):
          other = core.astensor(other)

       return other.space().ones()




# --- Sparse gradient ------------------------------------------------------- #

class SparseGrad(TensorLike, Pluggable):

   # --- Construction --- #

   def __init__(self, backend, shape, idxs, vals):

       self._backend = backend
       self._shape   = shape
       self._idxs    = idxs
       self._vals    = vals


   # --- Plugging into function calls --- #

   def pluginto(self, funcall):

       return self.todense().pluginto(funcall)


   # --- Gradient accumulation --- #

   def addto(self, other):

       if not other:
          other = ZeroGrad()

       if isinstance(other, ZeroGrad):
          other = self.space().zeros()

       if isinstance(other, SparseGrad):
          other = other.todense()

       data = self._backend.put(
                 other._data, self._idxs, self._vals, accumulate=True
              )

       return other.withdata(data)

       
   # --- Basic functionality --- #

   def copy(self):

       return self.__class__(
          self._backend, self._shape, self._idxs, self._vals
       )


   def todense(self):

       zeros = self._backend.zeros(self.shape, dtype=self.dtype)
       out   = self._backend.put(zeros, self._idxs, self._vals)

       return core.astensor(out, backend=self._backend)


   def withdata(self, data):

       return core.astensor(data, backend=self._backend)


   def space(self):

       return core.TensorSpace(
          self._backend.name(), self.shape, self.dtype
       )


   def item(self, *idx):

       return self.todense().item(*idx)  


   # --- Tensor properties --- #

   @property
   def dtype(self):
       return self._backend.dtype(self._vals) 

   @property 
   def size(self):
       return len(self._vals) 

   @property 
   def ndim(self):
       return len(self._shape) 

   @property 
   def shape(self):
       return self._shape


   # --- Comparisons --- #
       
   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          log.val(self._backend, other._backend)
          log.val(self._shape,   other._shape)
          log.val(self._idxs,    other._idxs)

       if bool(log):
          return self._backend.allequal(self._vals, other._vals)  

       return False


   # --- Arithmetics and element access --- # 

   def __getitem__(self, idx):

       return self.todense()[idx]


   def __neg__(self):

       return -self.todense()


   def __add__(self, other): 

       return self.todense() + other


   def __sub__(self, other): 

       return self.todense() - other   


   def __mul__(self, other):

       return self.todense() * other 


   def __truediv__(self, other):

       return self.todense() / other 


   def __pow__(self, other):

       return self.todense() ** other 


   def __radd__(self, other): 

       return other + self.todense() 


   def __rsub__(self, other): 

       return other - self.todense() 


   def __rmul__(self, other):

       return other * self.todense()


   def __rtruediv__(self, other):

       return other / self.todense()


   def __rpow__(self, other):

       return other ** self.todense() 




