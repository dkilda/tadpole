#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util     as util
import tadpole.index    as tid
import tadpole.autodiff as ad
import tadpole.array    as ar

import tadpole.tensor.core              as core
import tadpole.tensor.linalg.truncation as truncation


from tadpole.tensor.linalg.types import (
   CutoffMode,
   ErrorMode,
   Trunc,
)


from tadpole.tensor.linalg.truncation import (
   TruncNull,
)


from tadpole.tensor.types import (
   Tensor,
   Grad,
   Pluggable,
   Engine,
)


from tadpole.tensor.engine import (
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
###  Helpers: indexing and other logic needed for a tensor decomposition    ### 
###                                                                         ###
###############################################################################


# --- Functor that creates the s-index emerging from decomposition ---------- #

class SIndexFun:

   def __init__(self, name):

       self._name = name
       self._ind  = None


   def __call__(self, size):

       if self._ind is None:
          self._ind = IndexGen(self._name, size)

       if size != len(self._ind):
          raise ValueError(
             f"{type(self).__name__}: an attempt to resize link index to "
             f"an incompatible size {size} != original size {len(self._ind)}"
          )

       return self._ind




###############################################################################
###                                                                         ###
###  Tensor decomposition engine and operator                               ###
###                                                                         ###
###############################################################################


# --- Tensor decomposition factory ------------------------------------------ #

def tensor_decomp(x, sind=None):

    if sind is None:
       sind = "sind"

    if not isinstance(sind, SIndexFun):
       sind = SIndexFun(sind)

    engine = x.pluginto(EngineDecomp(slink))
    return engine.operator()




# --- Tensor decomposition engine ------------------------------------------- #

class EngineDecomp(Engine): 

   def __init__(self, sind, train=None):

       if train is None:
          train = TrainTensorData()

       self._sind  = sind
       self._train = train


   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          log.val(self._sind,  other._sind)
          log.val(self._train, other._train)

       return bool(log)


   @property
   def _size(self):

       return 1


   def attach(self, data, inds):

       if self._train.size() == self._size:
          raise TooManyArgsError(self, self._size)

       return self.__class__(self._sind, self._train.attach(data, inds))


   def operator(self):

       data, = self._train.data()
       inds, = self._train.inds()

       return TensorDecomp(data, inds, self._sind)




# --- Tensor decomposition operator ----------------------------------------- #

class TensorDecomp:

   # --- Construction --- #

   def __init__(self, data, inds, sind):

       self._data = data
       self._inds = inds
       self._sind = sind


   # --- Private helpers --- #

   def _ltensor(self, data):

       return core.TensorGen(data, (self._inds[0], self._sind(data.shape[1])))  


   def _stensor(self, data):

       return core.TensorGen(data, (self._sind(data.shape[0]), ))  


   def _rtensor(self, data):

       return core.TensorGen(data, (self._sind(data.shape[0]), self._inds[1]))  


   def _explicit(self, fun, trunc):

       output_data = fun(self._data)
       error       = trunc.error(output_data[1])
       output_data = trunc.apply(*output_data)

       return (
               self._ltensor(output_data[0]), 
               self._stensor(output_data[1]), 
               self._rtensor(output_data[2]), 
               error,
              )

       
   def _hidden(self, fun):

       output_data = fun(self._data)

       return (
               self._ltensor(output_data[0]), 
               self._rtensor(output_data[1]), 
              )


   # --- Explicit-rank decompositions --- #

   def svd(self, trunc):

       return self._explicit(ar.svd, trunc)


   def eig(self, trunc):

       return self._explicit(ar.eig, trunc)


   def eigh(self, trunc):

       return self._explicit(ar.eigh, trunc)


   # --- Hidden-rank decompositions --- #

   def qr(self):

       return self._hidden(ar.qr)


   def lq(self):

       return self._hidden(ar.lq)




###############################################################################
###                                                                         ###
###  Standalone functions corresponding to TensorDecomp methods             ###
###                                                                         ###
###############################################################################


# --- Explicit-rank decompositions ------------------------------------------ #

@ad.differentiable
def svd(x, sind=None, trunc=None):

    if trunc is None:
       trunc = TruncNull()

    op = tensor_decomp(x, sind)

    return op.svd(trunc)



@ad.differentiable
def eig(x, sind=None, trunc=None):

    if trunc is None:
       trunc = TruncNull()

    op = tensor_decomp(x, sind) 

    return op.eig(trunc)



@ad.differentiable
def eigh(x, sind=None, trunc=None):

    if trunc is None:
       trunc = TruncNull()

    op = tensor_decomp(x, sind) 

    return op.eigh(trunc)




# --- Hidden-rank decompositions -------------------------------------------- #

@ad.differentiable
def qr(x, sind=None):

    op = tensor_decomp(x, sind)

    return op.qr()



@ad.differentiable
def lq(x, sind=None):

    op = tensor_decomp(x, sind)

    return op.lq()




