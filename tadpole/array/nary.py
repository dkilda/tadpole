#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util as util

import operator
from functools import reduce

import tadpole.array.backends as backends
import tadpole.array.types    as types
import tadpole.array.void     as void
import tadpole.array.unary    as unary
import tadpole.array.binary   as binary




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


   # --- Array methods --- #

   def new(self, data):

       return unary.asarray(data, backend=self._backend) 


   def nary(self):

       return self


   def __or__(self, other):

       backend = backends.common(
          self._backend, 
          other._backend, 
          msg=f"{type(self).__name__}.__or__"
       )

       return self.__class__(backend, *self._datas, *other._datas)


   # --- Comparison --- #

   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          log.val(self._backend, other._backend)
 
       if bool(log):
          return all(self._backend.allequal(x, y) 
                        for x, y in zip(self._datas, other._datas))
               
       return False


   # --- Value methods --- #

   def where(self): 

       data = self._backend.where(*self._datas)

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
    array = array.nary()
            
    return array.einsum(equation, optimize=optimize)




