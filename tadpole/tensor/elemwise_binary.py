#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import functools

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.array    as ar
import tadpole.index    as tid

import tadpole.tensor.core        as core
import tadpole.tensor.reindexing  as reidx
import tadpole.tensor.interaction as tni


from tadpole.tensor.types import (
   Pluggable,
   Engine,
)


from tadpole.tensor.engine import (
   EngineUnary,
   EngineElemwise,
   TrainTensorData,
   TooManyArgsError,
)


from tadpole.index import (
   Index,
   IndexGen,  
   Indices,
)




###############################################################################
###                                                                         ###
###  Tensor binary elementwise engine and operator                          ###
###                                                                         ###
###############################################################################


# --- Tensor binary elementwise factory ------------------------------------- #

def tensor_elemwise_binary(x, y):

    engine = EngineElemwiseBinary()
    engine = x.pluginto(engine)
    engine = y.pluginto(engine)

    return engine.operator()




# --- Tensor binary elementwise engine -------------------------------------- #

class EngineElemwiseBinary(Engine):

   def __init__(self, source=None):

       if source is None:
          source = EngineElemwise(TensorElemwiseBinary, 2)

       self._source = source


   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          log.val(self._source, other._source)

       return bool(log)


   def attach(self, data, inds):

       return self.__class__(self._source.attach(data, inds))


   def operator(self):

       return self._source.operator()




# --- TensorElemwiseBinary operator ----------------------------------------- #

class TensorElemwiseBinary: 

   # --- Construction --- #

   def __init__(self, dataA, dataB, inds): 

       self._dataA = dataA
       self._dataB = dataB
       self._inds  = inds


   # --- Private helpers --- #

   def _apply(self, fun, *args, **kwargs):

       data = fun(self._dataA, self._dataB, *args, **kwargs)

       return core.TensorGen(data, self._inds)


   # --- Standard math --- #
 
   def add(self):

       return self._apply(ar.add)
 

   def sub(self):

       return self._apply(ar.sub)
 

   def mul(self):

       return self._apply(ar.mul)
 

   def div(self):

       return self._apply(ar.div)
 

   def mod(self):

       return self._apply(ar.mod)
 

   def floordiv(self):

       return self._apply(ar.floordiv)
 

   def power(self):

       return self._apply(ar.power)


   # --- Logical operations --- #

   def allclose(self, **opts):

       return ar.allclose(self._dataA, self._dataB, **opts)


   def isclose(self, **opts): 

       return self._apply(ar.isclose, **opts)


   def allequal(self):

       return ar.allequal(self._dataA, self._dataB)


   def isequal(self): 

       return self._apply(ar.isequal)


   def notequal(self): 

       return self._apply(ar.notequal)


   def greater(self): 

       return self._apply(ar.greater)


   def less(self): 

       return self._apply(ar.less)


   def greater_equal(self): 

       return self._apply(ar.greater_equal)


   def less_equal(self): 

       return self._apply(ar.less_equal)


   def logical_and(self): 

       return self._apply(ar.logical_and)


   def logical_or(self): 

       return self._apply(ar.logical_or)




###############################################################################
###                                                                         ###
###  Standalone functions corresponding to TensorElemwiseBinary methods     ###
###                                                                         ###
###############################################################################


# --- Helper: binary typecast ----------------------------------------------- #

def typecast_binary(fun):

    @functools.wraps(fun)
    def wrap(x, y, *args, **kwargs):

        try:
            return fun(x, y, *args, **kwargs)       
 
        except (AttributeError, TypeError):

            if not any(isinstance(v, Pluggable) for v in (x,y)):
               x = core.astensor(x)
               y = core.astensor(y) 

            if not isinstance(x, Pluggable):
               x = y.withdata(x) 

            if not isinstance(y, Pluggable):
               y = x.withdata(y) 

            return fun(x, y, *args, **kwargs)
         
    return wrap




# --- Gradient accumulation ------------------------------------------------- #

@ad.differentiable
@typecast_binary
def addgrads(x, y):

    return y.addto(x)




# --- Standard math --------------------------------------------------------- #

@ad.differentiable
@typecast_binary
def add(x, y):
 
    op = tensor_elemwise_binary(x, y)
    return op.add() 
 

@ad.differentiable
@typecast_binary
def sub(x, y):

    op = tensor_elemwise_binary(x, y)
    return op.sub() 
 

@ad.differentiable
@typecast_binary
def mul(x, y):

    op = tensor_elemwise_binary(x, y)
    return op.mul() 
 

@ad.differentiable
@typecast_binary
def div(x, y):

    op = tensor_elemwise_binary(x, y)
    return op.div() 
 

@ad.differentiable
@typecast_binary
def mod(x, y):

    op = tensor_elemwise_binary(x, y)
    return op.mod() 
 

@ad.nondifferentiable
@typecast_binary
def floordiv(x, y):

    op = tensor_elemwise_binary(x, y)
    return op.floordiv() 
 

@ad.differentiable
@typecast_binary
def power(x, y):

    op = tensor_elemwise_binary(x, y)
    return op.power() 




# --- Logical operations ---------------------------------------------------- #

@ad.nondifferentiable
@typecast_binary
def allclose(x, y, **opts):

    op = tensor_elemwise_binary(x, y)
    return op.allclose(**opts) 


@ad.nondifferentiable
@typecast_binary
def isclose(x, y, **opts): 

    op = tensor_elemwise_binary(x, y)
    return op.isclose(**opts) 


@ad.nondifferentiable
@typecast_binary
def allequal(x, y):

    op = tensor_elemwise_binary(x, y)
    return op.allequal() 


@ad.nondifferentiable
@typecast_binary
def isequal(x, y): 

    op = tensor_elemwise_binary(x, y)
    return op.isequal() 


@ad.nondifferentiable
@typecast_binary
def notequal(x, y): 

    op = tensor_elemwise_binary(x, y)
    return op.notequal() 


@ad.nondifferentiable
@typecast_binary
def greater(x, y): 

    op = tensor_elemwise_binary(x, y)
    return op.greater() 


@ad.nondifferentiable
@typecast_binary
def less(x, y): 

    op = tensor_elemwise_binary(x, y)
    return op.less() 


@ad.nondifferentiable
@typecast_binary
def greater_equal(x, y): 

    op = tensor_elemwise_binary(x, y)
    return op.greater_equal() 


@ad.nondifferentiable
@typecast_binary
def less_equal(x, y): 

    op = tensor_elemwise_binary(x, y)
    return op.less_equal()
 

@ad.nondifferentiable
@typecast_binary
def logical_and(x, y): 

    op = tensor_elemwise_binary(x, y)
    return op.logical_and() 


@ad.nondifferentiable
@typecast_binary
def logical_or(x, y): 

    op = tensor_elemwise_binary(x, y)
    return op.logical_or() 




