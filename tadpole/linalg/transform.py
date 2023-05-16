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
###  Linalg transform engine and operator                                   ###
###                                                                         ###
###############################################################################


# --- Linalg transform factory ---------------------------------------------- #

def linalg_transform(xs):

    engine = EngineLinalgTransform()
    for x in xs:
        engine = x.pluginto(engine)

    return engine.operator()




# --- Linalg transform engine ----------------------------------------------- #

class EngineLinalgTransform(tn.Engine):

   def __init__(self, train=None):

       if train is None:
          train = TrainTensorData()

       self._train = train


   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          log.val(self._train, other._train)

       return bool(log)


   def attach(self, data, inds):

       return self.__class__(self._train.attach(data, inds))


   def operator(self):

       return LinalgTransform(self._train.data(), self._train.inds())




# --- Linalg transform operator --------------------------------------------- #

class LinalgTransform:

   # --- Construction --- #

   def __init__(self, data, inds): 

       if all(x.ndim == 2 for x in inds):
          raise ValueError(
             f"LinalgTransform: input must have ndim = 2, but "
             f"data  ndims = {tuple(x.ndim for x in data)}, "
             f"index ndims = {tuple(x.ndim for x in inds)}."
          )

       self._data = data
       self._inds = inds


   # --- Linear algebra transformations --- #

   def lstack(self, inds, **opts):

       data = ar.stack(*self._data, axis=0, **opts)
       return tn.TensorGen(data, inds)


   def rstack(self, inds, **opts):

       data = ar.stack(*self._data, axis=1, **opts)
       return tn.TensorGen(data, inds)
       



###############################################################################
###                                                                         ###
###  Standalone functions corresponding to LinalgTransform methods          ###
###                                                                         ###
###############################################################################


# --- Linear algebra transformations ---------------------------------------- #

@ad.differentiable
def lstack(xs, inds, **opts):

    op = linalg_transform(xs)
    return op.lstack(inds, **opts)  


@ad.differentiable
def rstack(xs, inds, **opts):

    op = linalg_transform(xs)
    return op.rstack(inds, **opts)  




