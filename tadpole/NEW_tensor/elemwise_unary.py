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
###  Tensor elementwise unary operations                                    ###
###                                                                         ###
###############################################################################


# --- Elementwise unary engine: creates TensorElemwiseUnary ----------------- #

class EngineElemwiseUnary(Engine):

   def __init__(self, train=None):

       if train is None:
          train = TrainTensorData()

       self._train = train


   def size(self):
       
       return 1


   def attach(self, data, inds):

       if self._train.size() == self.size():
          raise TooManyArgsError(self, self.size())

       return self.__class__(self._train.attach(data, inds))


   def operator(self):

       data, = self._train.data()
       inds, = self._train.inds()

       return TensorElemwiseUnary(data, inds)




# --- Factory: creates TensorElemwiseUnary ---------------------------------- #

def tensor_elemwise_unary(x):

    engine = x.pluginto(EngineElemwiseUnary())
    return engine.operator()




# --- TensorElemwiseUnary operator ------------------------------------------ #

class TensorElemwiseUnary:

   # --- Construction --- #

   def __init__(self, data, inds): 

       self._data = data
       self._inds = inds


   # --- Private helpers --- #

   def _apply(fun, *args, **kwargs):

       data = fun(self._data, *args, **kwargs)

       return core.TensorGen(data, self._inds)


   # --- Value methods --- #

   def put(self, pos, vals, accumulate=False):

       return self._apply(ar.put, pos, vals, accumulate=accumulate)


   def clip(self, minval, maxval, **opts):

       return self._apply(ar.clip, minval, maxval, **opts)


   # --- Standard math --- #

   def neg(self):

       return self._apply(ar.neg)

   
   def sign(self):

       return self._apply(ar.sign)

 
   def conj(self):

       return self._apply(ar.conj)


   def real(self):

       return self._apply(ar.real)


   def imag(self):

       return self._apply(ar.imag)


   def absolute(self):

       return self._apply(ar.absolute)


   def sqrt(self):

       return self._apply(ar.sqrt)


   def log(self):

       return self._apply(ar.log)


   def exp(self):

       return self._apply(ar.exp)


   def sin(self):

       return self._apply(ar.sin)


   def cos(self):

       return self._apply(ar.cos)


   def tan(self):

       return self._apply(ar.tan)


   def arcsin(self):

       return self._apply(ar.arcsin)


   def arccos(self):

       return self._apply(ar.arccos)


   def arctan(self):

       return self._apply(ar.arctan)


   def sinh(self):

       return self._apply(ar.sinh)


   def cosh(self):

       return self._apply(ar.cosh)


   def tanh(self):

       return self._apply(ar.tanh)


   def arcsinh(self):

       return self._apply(ar.arcsinh)


   def arccosh(self):

       return self._apply(ar.arccosh)


   def arctanh(self):

       return self._apply(ar.arctanh)


   # --- Linear algebra --- #

   def expm(self):

       return self._apply(ar.expm)







 


###############################################################################
###                                                                         ###
###  Standalone functions corresponding to TensorElemwiseUnary methods      ###
###                                                                         ###
###############################################################################


# --- Value methods --------------------------------------------------------- #



# --- Standard math --------------------------------------------------------- #



# --- Linear algebra -------------------------------------------------------- #







































