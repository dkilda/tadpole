#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import functools

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.array    as ar
import tadpole.index    as tid

import tadpole.tensor.core as core


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


from tadpole.tensor.elemwise_binary import (
   typecast_binary,
)


from tadpole.index import (
   Index,
   IndexGen, 
   Indices,
)




###############################################################################
###                                                                         ###
###  Tensor ternary elementwise engine and operator                         ###
###                                                                         ###
###############################################################################


# --- Tensor ternary elementwise factory ------------------------------------ #

def tensor_elemwise_ternary(x, y, z):

    engine = EngineElemwiseTernary()
    engine = x.pluginto(engine)
    engine = y.pluginto(engine)
    engine = z.pluginto(engine)

    return engine.operator()




# --- Tensor ternary elementwise engine ------------------------------------- #

class EngineElemwiseTernary(Engine):

   def __init__(self, source=None):

       if source is None:
          source = EngineElemwise(TensorElemwiseTernary, 3)

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




# --- Tensor ternary elementwise operator ----------------------------------- #

class TensorElemwiseTernary: 

   # --- Construction --- #

   def __init__(self, dataA, dataB, dataC, inds): 

       self._dataA = dataA
       self._dataB = dataB
       self._dataC = dataC
       self._inds  = inds


   # --- Private helpers --- #

   def _apply(self, fun, *args, **kwargs):

       data = fun(self._dataA, self._dataB, self._dataC, *args, **kwargs)

       return core.TensorGen(data, self._inds)


   # --- Value methods --- #

   def where(self):

       return self._apply(ar.where)




###############################################################################
###                                                                         ###
###  Standalone functions corresponding to TensorElemwiseTernary methods    ###
###                                                                         ###
###############################################################################


# --- Helper: binary typecast ----------------------------------------------- #

def typecast_ternary(fun):

    @functools.wraps(fun)
    def wrap(x, y, z, *args, **kwargs):

        try:
            return fun(x, y, z, *args, **kwargs)       
 
        except (AttributeError, TypeError):

            pluggables = [v for v in (x,y,z) if isinstance(v, Pluggable)]

            if len(pluggables) == 0:
               x = core.astensor(x)
               y = core.astensor(y) 
               z = core.astensor(z) 

            if not isinstance(x, Pluggable):
               x = pluggables[0].withdata(x) 

            if not isinstance(y, Pluggable):
               y = pluggables[0].withdata(y) 

            if not isinstance(z, Pluggable):
               z = pluggables[0].withdata(z) 

            return fun(x, y, z, *args, **kwargs)
         
    return wrap


# --- Value methods --------------------------------------------------------- #

@ad.differentiable
@typecast_ternary
def where(condition, x, y):

    op = tensor_elemwise_ternary(condition, x, y)

    return op.where()



"""
@ad.differentiable
def where(condition, x, y):

    print("WHERE: ", condition, x, y)

    @typecast_binary
    def fun(x, y):

        op = tensor_elemwise_ternary(condition, x, y)
        return op.where()

    return fun(x, y)
"""



