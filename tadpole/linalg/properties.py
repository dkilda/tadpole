#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.array    as ar
import tadpole.tensor   as tn
import tadpole.index    as tid

from tadpole.tensor.engine import (
   EngineUnary,
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
###  Linalg properties engine and operator                                  ###
###                                                                         ###
###############################################################################


# --- Linalg property factory ----------------------------------------------- #

def linalg_properties(x):

    engine = x.pluginto(EngineLinalgProperties())
    return engine.operator()




# --- Linalg property engine ------------------------------------------------ #

class EngineLinalgProperties(tn.Engine):

   def __init__(self, source=None):

       if source is None:
          source = EngineUnary(LinalgProperties)

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




# --- Linalg properties operator -------------------------------------------- #

class LinalgProperties:

   # --- Construction --- #

   def __init__(self, data, inds): 

       if inds.ndim != 2:
          raise ValueError(
             f"LinalgProperties: input must have ndim = 2, "
             f"but data.ndim = {data.ndim}, inds.ndim = {inds.ndim}."
          )

       self._data = data
       self._inds = inds


   # --- Private helpers --- #

   def _apply(self, fun, *args, **kwargs):

       data = fun(self._data, *args, **kwargs) 
       return tn.TensorGen(data, Indices()) 


   # --- Linear algebra properties --- #

   def norm(self, order=None, **opts):

       return self._apply(ar.norm, order=order, **opts)


   def trace(self, **opts):

       return self._apply(ar.trace, **opts)
      

   def det(self): 

       return self._apply(ar.det)


   def inv(self): 

       return self._apply(ar.inv)




###############################################################################
###                                                                         ###
###  Standalone functions corresponding to LinalgProperties methods         ###
###                                                                         ###
###############################################################################


# --- Linear algebra properties --------------------------------------------- #

@ad.differentiable
def norm(x, order=None, **opts):

    op = linalg_properties(x)
    return op.norm(order, **opts)  


@ad.differentiable
def trace(x, **opts):

    op = linalg_properties(x)
    return op.trace(**opts) 


@ad.differentiable
def det(x):

    op = linalg_properties(x)
    return op.det() 


@ad.differentiable
def inv(x):

    op = linalg_properties(x)
    return op.inv() 




