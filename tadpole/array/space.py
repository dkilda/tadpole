#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import tadpole.util as util

import tadpole.array.backends as backends
import tadpole.array.types    as types
import tadpole.array.void     as void
import tadpole.array.unary    as unary
import tadpole.array.binary   as binary



###############################################################################
###                                                                         ###
###  Definition of Array Space                                              ###
###                                                                         ###
###############################################################################


# --- Create Array Space ---------------------------------------------------- #

def arrayspace(shape, dtype=None, backend=None):

    backend = backends.get(backend)
    dtype   = backend.get_dtype(dtype)

    return ArraySpace(backend, shape, dtype)




# --- Array Space ----------------------------------------------------------- #

class ArraySpace(types.Space): 

   # --- Construction --- #

   def __init__(self, backend, shape, dtype):

       self._backend = backend
       self._shape   = shape
       self._dtype   = dtype


   # --- Private helpers --- #

   @property
   def _void(self):

       return void.Array(self._backend)


   # --- Comparisons --- #

   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          log.val(self._backend, other._backend)
          log.val(self._shape,   other._shape)
          log.val(self._dtype,   other._dtype)

       return bool(log)


   # --- Fill space with data --- #

   def fillwith(self, data):

       def iscomplex(dtype):
           return dtype in self._backend.complex_dtypes()

       data = unary.asarray(data, backend=self._backend)

       if data.shape != self.shape:
          data = unary.broadcast_to(data, self.shape)

       if unary.iscomplex(data) and not iscomplex(self.dtype):
          return data

       return unary.astype(data, dtype=self.dtype)


   # --- Reshape space --- #

   def reshape(self, shape):

       return self.__class__(self._backend, shape, self._dtype)


   # --- Space properties --- #

   @property
   def dtype(self):
       return self._dtype

   @property
   def size(self):
       return int(np.prod(self._shape)) 

   @property 
   def ndim(self):
       return len(self._shape)

   @property
   def shape(self):
       return self._shape


   # --- Array creation methods --- #

   def zeros(self, **opts):

       return self._void.zeros(
          self._shape, dtype=self._dtype, **opts
       )


   def ones(self, **opts):

       return self._void.ones(
          self._shape, dtype=self._dtype, **opts
       )


   def unit(self, idx, **opts):

       return self._void.unit(
          self._shape, idx, dtype=self._dtype, **opts
       )


   def eye(self, N, M=None, **opts):

       return self._void.eye(
          N, M=M, dtype=self._dtype, **opts
       )


   def rand(self, **opts):

       return self._void.rand(
          self._shape, dtype=self._dtype, **opts
       )


   def randn(self, **opts):

       return self._void.randn(
          self._shape, dtype=self._dtype, **opts
       )


   def randuniform(self, boundaries, **opts):

       return self._void.randuniform(
          self._shape, boundaries, dtype=self._dtype, **opts
       )


   # --- Array generators --- #

   def units(self, **opts):

       for idx in np.ndindex(*self._shape):
           yield self.unit(idx, **opts)


   def basis(self, **opts):

       if  self._void.iscomplex_type(self._dtype):

           for unit in self.units(**opts):
               yield unit
               yield binary.mul(1j, unit)

       else:
           for unit in self.units(**opts):
               yield unit





