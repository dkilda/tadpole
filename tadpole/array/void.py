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
###  Misc methods                                                           ###
###                                                                         ###
###############################################################################


# --- Void Array factory ---------------------------------------------------- #

def asvoid(backend=None):

    return Array(backends.get(backend))




# --- Decorator that creates a Void Array with an appropriate backend ------- #

def fromvoid(fun):

    def wrap(*args, **opts):

        x = Array(backends.get_from(opts))
        return fun(x, *args, **opts)

    return wrap




###############################################################################
###                                                                         ###
###  Definition of Void Array (supports array creation)                     ###
###                                                                         ###
###############################################################################


# --- Void Array ------------------------------------------------------------ #

class Array(ArrayLike): # TODO should we include dtype, shape in there? so that it becomes more like space?

   # --- Construction --- #

   def __init__(self, backend):

       if not isinstance(backend, backends.Backend):
          raise ValueError(
             f"VoidArray: backend must be an instance "
             f"of Backend, but it is {backend}"
          ) 

       self._backend = backend


   # --- Arraylike methods --- #

   def new(self, data):

       return unary.OneArray(self._backend, data)


   def __or__(self, other):

       return other 


   # --- Data type methods  --- #
 
   def iscomplex_type(self, dtype):

       dtype = self._backend.get_dtype(dtype)

       return dtype in self._backend.complex_dtypes()


   def get_dtype(self, dtype=None):

       return self._backend.get_dtype(dtype)


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


# --- Data type methods ----------------------------------------------------- #

@fromvoid 
def iscomplex_type(x, dtype):

    return x.iscomplex_type(dtype)


@fromvoid 
def get_dtype(x, dtype=None):

    return x.get_dtype(dtype)




# --- Array creation methods ------------------------------------------------ #
    
@fromvoid
def zeros(x, shape, **opts):

    return x.zeros(shape, **opts)


@fromvoid
def ones(x, shape, **opts):

    return x.ones(shape, **opts)


@fromvoid
def unit(x, shape, idx, **opts):

    return x.unit(shape, idx, **opts)


@fromvoid
def eye(x, N, M=None, **opts):

    return x.eye(N, M=None, **opts)


@fromvoid
def rand(x, shape, **opts):

    return x.rand(shape, **opts)


@fromvoid
def randn(x, shape, **opts):

    return x.randn(shape, **opts)
       

@fromvoid
def randuniform(x, shape, boundaries, **opts):

    return x.randn(shape, boundaries, **opts)




