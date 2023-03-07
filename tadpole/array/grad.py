#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np

import tadpole.array.operations as op
import tadpole.array.function   as function
import tadpole.array.backends   as backends
import tadpole.array.core       as core
import tadpole.autodiff         as ad
import tadpole.util             as util

from tadpole.array.arraylike import Pluggable, ArrayLike
from tadpole.array.function  import Args, TransformCall




###############################################################################
###                                                                         ###
###  A general framework for array gradients                                ###
###                                                                         ###
###############################################################################


# --- Zero gradient --------------------------------------------------------- #

class ZeroGrad(ArrayLike, Pluggable):

   # --- Construction --- #

   def __init__(self):

       self._backend = backends.get(None)


   # --- Convert to dense --- #

   def todense(self):

       return self.space().zeros()


   # --- Plugging into function calls --- #

   def pluginto(self, funcall):

       return self.todense().pluginto(funcall)


   # --- Using in gradient accumulations --- #

   def addto(self, other):

       return other


   # --- Basic functionality --- #

   def copy(self):

       return self.__class__()


   def asarray(self, data):

       return core.asarray(self._backend, data)


   def space(self):

       return core.ArraySpace(
                              self._backend.name(), 
                              tuple(), 
                              self._backend.get_dtype(None)
                             )
   def item(self):

       return self.todense().item() 


   # --- Array properties --- #

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

   def allclose(self, other, **opts):

       if not isinstance(other, self.__class__):
          return self.todense().allclose(other, **opts) 

       log = util.LogicalChain()
       log.typ(self, other)

       return bool(log)

       
   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       return bool(log)


   # --- Arithmetics and element access --- #

   def __getitem__(self, idx):

       return self.todense()[idx]


   def __neg__(self):

       return -self.todense()


   def __add__(self, other): 

       return other


   def __sub__(self, other): 

       return -other 


   def __mul__(self, other):

       return self.todense().item()


   def __radd__(self, other): 

       return self.__add__(other)  


   def __rsub__(self, other): 

       return -self.__sub__(other)  


   def __rmul__(self, other):

       return self.__mul__(other)




# --- Sparse gradient ------------------------------------------------------- #

class SparseGrad(ArrayLike, Pluggable):

   # --- Construction --- #

   def __init__(self, backend, shape, idxs, vals):

       self._backend = backend
       self._shape   = shape
       self._idxs    = idxs
       self._vals    = vals


   # --- Convert to dense --- #

   def todense(self):

       return op.put(self.space().zeros(), self._idxs, self._vals)


   # --- Plugging into function calls --- #

   def pluginto(self, funcall):

       return self.todense().pluginto(funcall)


   # --- Using in gradient accumulations --- #

   def addto(self, other):

       if isinstance(other, ZeroGrad):
          other = self.space().zeros()

       if isinstance(other, SparseGrad):
          other = other.todense()

       data = self._backend.put(
                 other._data, self._idxs, self._vals, accumulate=True
              )

       return other.asarray(data)

       
   # --- Basic functionality --- #

   def copy(self):

       return self.__class__(
          self._backend, self._shape, self._idxs, self._vals
       )


   def asarray(self, data):

       return core.asarray(data, backend=self._backend)


   def space(self):

       return core.ArraySpace(
          self._backend.name(), self.shape, self.dtype
       )


   def item(self, *idx):

       return self.todense().item(*idx)  


   # --- Array properties --- #

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

   def allclose(self, other, **opts):

       if not isinstance(other, self.__class__):
          return self.todense().allclose(other, **opts) 

       log = util.LogicalChain()
       log.typ(self, other)

       log.val(self._backend, other._backend)
       log.val(self._shape,   other._shape)
       log.val(self._idxs,    other._idxs)

       if bool(log):
          return util.allclose(self._vals, other._vals, **opts)   

       return False

       
   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          log.val(self._backend, other._backend)
          log.val(self._shape,   other._shape)
          log.val(self._idxs,    other._idxs)

       if bool(log):
          return util.allequal(self._vals, other._vals)  

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


   def __radd__(self, other): 

       return self.__add__(other)  


   def __rsub__(self, other): 

       return -self.__sub__(other)  


   def __rmul__(self, other):

       return self.__mul__(other)




