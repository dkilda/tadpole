#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.array    as ar
import tadpole.index    as tid

import tadpole.tensor.core as core


from tadpole.tensor.types import (
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
###  Tensor reduction engine and operator                                   ###
###                                                                         ###
###############################################################################


# --- Tensor reduction factory ---------------------------------------------- #

def tensor_reduce(x):

    engine = x.pluginto(EngineReduce())
    return engine.operator()




# --- Tensor reduction engine ----------------------------------------------- #

class EngineReduce(Engine):

   def __init__(self, source=None):

       if source is None:
          source = EngineUnary(TensorReduce)

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




# --- Tensor reduction operator --------------------------------------------- #

class TensorReduce:

   # --- Construction --- #

   def __init__(self, data, inds): 

       self._data = data
       self._inds = inds


   # --- Private helpers --- #

   def _map(self, *inds):

       return self._inds.map(*inds) 


   def _axes(self, *inds):

       return self._inds.axes(*inds)       


   def _output_inds(self, inds):

       return self._inds.remove(*inds)


   def _apply(self, fun, inds=None, **opts):

       if inds is None:
          data = fun(self._data, **opts)
          return core.TensorGen(data, Indices())

       inds = self._map(*inds)
       axes = self._axes(*inds)

       if len(axes) == 1:
          axes, = axes   

       data = fun(self._data, axes, **opts)

       return core.TensorGen(data, self._output_inds(inds))

      
   # --- Value methods --- #

   def allof(self, inds=None, **opts):

       return self._apply(ar.allof, inds, **opts)


   def anyof(self, inds=None, **opts):

       return self._apply(ar.anyof, inds, **opts)


   def amax(self, inds=None, **opts):

       return self._apply(ar.amax, inds, **opts)


   def amin(self, inds=None, **opts):

       return self._apply(ar.amin, inds, **opts)


   def count_nonzero(self, inds=None, **opts):

       return self._apply(ar.count_nonzero, inds, **opts)


   # --- Shape methods --- #

   def sumover(self, inds=None, dtype=None, **opts):

       return self._apply(ar.sumover, inds, dtype, **opts)


   def cumsum(self, ind=None, dtype=None, **opts):

       return self._apply(ar.cumsum, ind, dtype, **opts)


   # --- Linear algebra methods --- #

   def norm(self, inds=None, order=None, **opts):

       return self._apply(ar.norm, inds, order, **opts)

 


###############################################################################
###                                                                         ###
###  Standalone functions corresponding to TensorReduce methods             ###
###                                                                         ###
###############################################################################


# --- Value methods --------------------------------------------------------- #

@ad.differentiable
def allof(x, inds=None, **opts):

    op = tensor_reduce(x)
    return op.allof(inds, **opts)  


@ad.differentiable
def anyof(x, inds=None, **opts):

    op = tensor_reduce(x)
    return op.anyof(inds, **opts)  


@ad.differentiable
def amax(x, inds=None, **opts):

    op = tensor_reduce(x)
    return op.amax(inds, **opts)  


@ad.differentiable
def amin(x, inds=None, **opts):

    op = tensor_reduce(x)
    return op.amin(inds, **opts)  


@ad.differentiable
def count_nonzero(x, inds=None, **opts):

    op = tensor_reduce(x)
    return op.count_nonzero(inds, **opts)  




# --- Shape methods --------------------------------------------------------- #

@ad.differentiable
def sumover(x, inds=None, dtype=None, **opts):

    op = tensor_reduce(x)
    return op.sumover(inds, dtype, **opts)  


@ad.differentiable
def cumsum(x, ind=None, dtype=None, **opts):

    op = tensor_reduce(x)
    return op.cumsum(ind, dtype, **opts)




# --- Linear algebra methods ------------------------------------------------ #

@ad.differentiable
def norm(x, inds=None, order=None, **opts):

    op = tensor_reduce(x)
    return op.norm(inds, order, **opts)  






