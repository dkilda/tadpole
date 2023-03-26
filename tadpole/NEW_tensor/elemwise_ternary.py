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






































