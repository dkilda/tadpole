#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.array    as ar

import tadpole.tensor.core as core


from tadpole.tensor.types import (
   Engine,
)


from tadpole.tensor.engine import (
   EngineElemwise,
   TrainTensorData,
   TooManyArgsError,
)


from tadpole.tensor.index import (
   Index, 
   Indices,
   shapeof, 
   sizeof,
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

   def _apply(fun, *args, **kwargs):

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


# --- Value methods --------------------------------------------------------- #

@ad.differentiable
def where(condition, x, y):

    op = tensor_elemwise_ternary(condition, x, y)
    return op.where()



























###############################################################################
###                                                                         ###
###  Tensor elementwise ternary operations                                  ###
###                                                                         ###
###############################################################################


# --- Elementwise ternary engine: creates TensorElemwiseTernary ------------- #

class EngineElemwiseTernary(Engine):

   def __init__(self, train=None):

       if train is None:
          train = TrainTensorData()

       self._train = train


   def size(self):
       
       return 3


   def attach(self, data, inds):

       if self._train.size() == self.size():
          raise TooManyArgsError(self, self.size())

       return self.__class__(self._train.attach(data, inds))


   def operator(self):

       data = tuple(self._train.data())[:self.size()]
       inds = tuple(self._train.inds())[:self.size()]

       return TensorElemwiseTernary(data, inds)




# --- Factory: creates TensorElemwiseTernary -------------------------------- #

def tensor_elemwise_ternary(x):

    engine = x.pluginto(EngineElemwiseTernary())
    return engine.operator()





# --- TensorElemwiseTernary operations -------------------------------------- #




 


###############################################################################
###                                                                         ###
###  Standalone functions corresponding to TensorElemwiseTernary methods    ###
###                                                                         ###
###############################################################################


# --- TensorElemwiseTernary ------------------------------------------------- #






































