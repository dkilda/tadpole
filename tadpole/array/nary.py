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


   # --- Shape methods --- #

   def concat(self, axis=None, **opts):

       data = self._backend.concat(tuple(self._datas), axis=axis, **opts)

       return self.new(data) 


   # --- Value methods --- #

   def where(self): 

       data = self._backend.where(*self._datas)

       return self.new(data)


   def put(self, accumulate=False):

       idxs = tuple(map(self._backend.asarray, self._datas[1:-1]))
       vals = self._backend.asarray(self._datas[-1])

       data = self._backend.put(
                 self._datas[0], idxs, vals, accumulate=accumulate
              )

       return self.new(data) 


   # --- Contraction --- #

   def einsum(self, equation, optimize=True):

       data = self._backend.einsum(equation, *self._datas, optimize=optimize)

       return self.new(data)
 



###############################################################################
###                                                                         ###
###  Standalone functions corresponding to Nary Array methods               ###
###                                                                         ###
###############################################################################


# --- Shape methods --------------------------------------------------------- #

def concat(xs, axis=None, **opts):

    array = reduce(operator.or_, xs)
    array = array.nary()

    return array.concat(axis=axis, **opts)




# --- Value methods --------------------------------------------------------- #

def where(condition, x, y): 

    return (condition | x | y).where()



def put(x, idxs, vals, accumulate=False):

    idxs = tuple(map(unary.asarray, idxs))
    vals = unary.asarray(vals)

    array = x | reduce(operator.or_, idxs) | vals
    array = array.nary()

    return array.put(accumulate=accumulate)




# --- Contraction ----------------------------------------------------------- #

def einsum(equation, *xs, optimize=True):

    array = reduce(operator.or_, xs) 
    array = array.nary()
            
    return array.einsum(equation, optimize=optimize)




