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

   def __init__(self, inds=None):

       if inds is None:
          inds = index.Indices()

       self._backend = backends.get(None)
       self._inds    = inds


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
                              self._inds, 
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

   def __getitem__(self, pos):

       return self.todense()[pos]


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

   def __init__(self, backend, inds, pos, vals):

       self._backend = backend
       self._inds    = inds
       self._pos     = pos
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

       assert self._inds == other._inds, (
          f"SparseGrad.addto(): "
          f"gradient accumulation cannot be performed for tensors "
          f"with non-matching indices {self._inds} != {other._inds}"
       )

       data = self._backend.put(
                 other._data, self._pos, self._vals, accumulate=True
              )

       return other.withdata(data)

       
   # --- Basic functionality --- #

   def copy(self):

       return self.__class__(
          self._backend, self._shape, self._pos, self._vals
       )


   def todense(self):

       zeros = self._backend.zeros(self.shape, dtype=self.dtype)
       data  = self._backend.put(zeros, self._pos, self._vals)

       return core.astensor(data, self._inds, backend=self._backend)


   def withdata(self, data):

       return core.astensor(data, self._inds, backend=self._backend)


   def space(self):

       return core.TensorSpace(
          self._backend.name(), self._inds, self.dtype
       )


   def item(self, *pos):

       return self.todense().item(*pos)  


   # --- Tensor properties --- #

   @property
   def dtype(self):
       return self._backend.dtype(self._vals) 

   @property 
   def size(self):
       return self._inds.size

   @property 
   def ndim(self):
       return self._inds.ndim 

   @property 
   def shape(self):
       return self._inds.shape


   # --- Comparisons --- #
       
   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          log.val(self._backend, other._backend)
          log.val(self._inds,    other._inds)
          log.val(self._pos,     other._pos)

       if bool(log):
          return self._backend.allequal(self._vals, other._vals)  

       return False


   # --- Arithmetics and element access --- # 

   def __getitem__(self, pos):

       return self.todense()[pos]


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




