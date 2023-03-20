#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import tadpole.util     as util
import tadpole.backends as backends

import tadpole.array.core  as core
import tadpole.array.unary as unary

from tadpole.array.core import ArrayLike




###############################################################################
###                                                                         ###
###  Definition of Void Array (supports array creation)                     ###
###                                                                         ###
###############################################################################


# --- Void Array ------------------------------------------------------------ #

class Array(ArrayLike): 

   # --- Construction --- #

   def __init__(self, backend):

       if not isinstance(backend, backends.Backend):
          raise ValueError(
             f"{type(self).__name__}: "
             f"backend must be an instance of Backend, but it is {backend}"
          ) 

       self._backend = backend


   # --- Arraylike methods --- #

   def new(self, data):

       return unary.Array(self._backend, data)


   def __or__(self, other):

       return other 


   # --- Data type methods  --- #
 
   def iscomplex_type(self, dtype):

       dtype = self._backend.get_dtype(dtype)

       return dtype in self._backend.complex_dtypes()


   # --- Array creation methods --- #

   def zeros(self, shape, **opts):

       data = self._backend.zeros(shape, **opts)

       return self.new(data)


   def ones(self, shape, **opts):

       data = self._backend.ones(shape, **opts)     

       return self.new(data)     


   def unit(self, shape, idx, **opts):

       data = self._backend.unit(shape, idx, **opts)

       return self.new(data)
       

   def eye(self, N, M=None, **opts):

       data = self._backend.eye(N, M, **opts)

       return self.new(data)
       

   def rand(self, shape, **opts):

       data = self._backend.rand(shape, **opts)

       return self.new(data)
       

   def randn(self, shape, **opts):

       data = self._backend.randn(shape, **opts)

       return self.new(data)
       

   def randuniform(self, shape, boundaries, **opts):

       data = self._backend.randuniform(shape, boundaries, **opts)

       return self.new(data)




###############################################################################
###                                                                         ###
###  Standalone functions corresponding to Void Array methods               ###
###                                                                         ###
###############################################################################


# --- Automatic creation of VoidArray for Array factories ------------------- #

def auto_void(fun):

    def wrap(*args, **opts):

        x = Array(backends.get_from(opts))
        return fun(x, *args, **opts)

    return wrap




# --- Data type methods ----------------------------------------------------- #

@auto_void
def iscomplex_type(x, dtype, **opts):

    return x.iscomplex_type(dtype)




# --- Array creation methods ------------------------------------------------ #
    
@auto_void
def zeros(x, shape, **opts):

    return x.zeros(shape, **opts)


@auto_void
def ones(x, shape, **opts):

    return x.ones(shape, **opts)


@auto_void
def unit(x, shape, idx, **opts):

    return x.unit(shape, idx, **opts)


@auto_void
def eye(x, N, M=None, **opts):

    return x.eye(N, M=None, **opts)


@auto_void
def rand(x, shape, **opts):

    return x.rand(shape, **opts)


@auto_void
def randn(x, shape, **opts):

    return x.randn(shape, **opts)
       

@auto_void
def randuniform(x, shape, boundaries, **opts):

    return x.randn(shape, boundaries, **opts)




