#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.array    as ar


from tadpole.tensor.train import (
   TrainTensorData,
   TooManyArgsError,
)


from tadpole.tensor.types import (
   Engine,
)


from tadpole.tensor.index import (
   Index, 
   Indices,
   shapeof, 
   sizeof,
)




###############################################################################
###                                                                         ###
###  Tensor elementwise binary operations                                   ###
###                                                                         ###
###############################################################################


# --- Factory: creates TensorElemwiseBinary --------------------------------- #

def tensor_elemwise_binary(x, y):

    engine = EngineElemwiseBinary()
    engine = x.pluginto(engine)
    engine = y.pluginto(engine)

    return engine.operator()




# --- Elementwise binary engine: creates TensorElemwiseBinary --------------- #

class EngineElemwiseBinary(Engine): # TODO make separate templates for unary/binary/nary?

   def __init__(self, train=None):

       if train is None:
          train = TrainTensorData()

       self._train = train


   def size(self):
       
       return 2


   def attach(self, data, inds):

       if self._train.size() == self.size():
          raise TooManyArgsError(self, self.size())

       return self.__class__(self._train.attach(data, inds))


   def operator(self):

       data = self._train.data()
       inds = max(self._train.inds(), key=len)

       assert set(util.concat(self._train.inds())) == set(inds), (
          f"\n{type(self).__name__}.operator(): "
          f"An elementwise operation cannot be performed for tensors"
          f"with incompatible indices {tuple(self._train.inds())}."
       )

       return TensorElemwiseBinary(*data, inds)




# --- TensorElemwiseBinary operator ----------------------------------------- #

class TensorElemwiseBinary: 

   # --- Construction --- #

   def __init__(self, dataA, dataB, inds): 

       self._dataA = dataA
       self._dataB = dataB
       self._inds  = inds


   # --- Private helpers --- #

   def _apply(fun, *args, **kwargs):

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
 

   def power(self):

       return self._apply(ar.power)


   # --- Logical operations --- #

   def allclose(self, **opts):

       return self._apply(ar.allclose, **opts)


   def isclose(self, **opts): 

       return self._apply(ar.isclose, **opts)


   def allequal(self):

       return self._apply(ar.allequal)


   def isequal(self): 

       return self._apply(ar.isequal)


   def logical_and(self): 

       return self._apply(ar.logical_and)


   def logical_or(self): 

       return self._apply(ar.logical_or)




###############################################################################
###                                                                         ###
###  Standalone functions corresponding to TensorElemwiseBinary methods     ###
###                                                                         ###
###############################################################################


# --- Standard math --------------------------------------------------------- #



# --- Logical operations ---------------------------------------------------- #































