#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util     as util
import tadpole.backends as backends

import operator
from functools import reduce

import tadpole.array.types  as types
import tadpole.array.unary  as unary
import tadpole.array.binary as binary




###############################################################################
###                                                                         ###
###  Definition of Nary Array (supports nary operations)                    ###
###                                                                         ###
###############################################################################


# --- Nary Array ------------------------------------------------------------ #

class Array(types.Array):

   # --- Construction --- #

   def __init__(self, backend, *datas):

       if not isinstance(backend, backends.Backend):
          raise ValueError(
             f"{type(self).__name__}: "
             f"backend must be an instance of Backend, but it is {backend}"
          ) 

       self._backend = backend
       self._datas   = datas


   # --- Arraylike methods --- #

   def new(self, data):

       return unary.Array(self._backend, data)


   def __or__(self, other):

       backend = backends.common(
          self._backend, 
          other._backend, 
          msg=f"{type(self).__name__}.__or__"
       )

       return self.__class__(backend, *self._datas, *other._datas)


   # --- Value methods --- #

   def where(self): 

       data = self._backend.where(*self.datas)

       return self.new(data)


   # --- Linear algebra: products --- #

   def einsum(self, equation, optimize=True):

       data = self._backend.einsum(equation, *self._datas, optimize=optimize)

       return self.new(data)




###############################################################################
###                                                                         ###
###  Standalone functions corresponding to Nary Array methods               ###
###                                                                         ###
###############################################################################


# --- Value methods --------------------------------------------------------- #

def where(condition, x, y): 

    return (condition | x | y).where()




# --- Linear algebra: products ---------------------------------------------- #

def einsum(equation, *xs, optimize=True):

    array = reduce(operator.or_, xs) 
            
    return array.einsum(equation, optimize=optimize)




