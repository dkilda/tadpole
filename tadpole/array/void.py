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
             f"VoidArray: backend must be an instance "
             f"of Backend, but it is {backend}"
          ) 

       self._backend = backend


   # --- Arraylike methods --- #

   def new(self, data):

       return unary.OneArray(self._backend, data)


   def __or__(self, other):

       return other 


   # --- Array creation methods --- #

   def asarray(self, array, **opts):

       if isinstance(array, ArrayLike):
          return array

       data = self._backend.asarray(array, **opts)

       return self.new(data)


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


# --- Array creation methods ------------------------------------------------ #

def asarray(x, array, **opts):

    return x.asarray(array, **opts)


def zeros(x, shape, **opts):

    return x.zeros(shape, **opts)


def ones(x, shape, **opts):

    return x.ones(shape, **opts)


def unit(x, shape, idx, **opts):

    return x.unit(shape, idx, **opts)


def eye(x, N, M=None, **opts):

    return x.eye(N, M=None, **opts)


def rand(x, shape, **opts):

    return x.rand(shape, **opts)


def randn(x, shape, **opts):

    return x.randn(shape, **opts)
       

def randuniform(x, shape, boundaries, **opts):

    return x.randn(shape, boundaries, **opts)




