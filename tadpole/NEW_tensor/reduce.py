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
###  Tensor reduction operations                                            ###
###                                                                         ###
###############################################################################


# --- Reduction engine: creates TensorReduce -------------------------------- #

class EngineReduce(Engine):

   def __init__(self, train=None):

       if train is None:
          train = TrainTensorData()

       self._train = train


   def attach(self, data, inds):

       if self._train.size() == 1:
          raise TooManyArgsError(self, 1)

       return self.__class__(self._train.attach(data, inds))


   def operator(self):

       data, = self._train.data()
       inds, = self._train.inds()

       return TensorReduce(data, inds)




# --- Factory: creates TensorReduce ----------------------------------------- #

def tensor_reduce(x):

    engine = x.pluginto(EngineReduce())
    return engine.operator()


    

# --- TensorReduce operator ------------------------------------------------- #

class TensorReduce:

   # --- Construction --- #

   def __init__(self, data, inds): 

       self._data = data
       self._inds = inds


   # --- Private helpers --- #

   def _map(self, inds):

       if isinstance(inds, (str, Index)):
          inds = (inds, )

       return self._inds.map(*inds) 


   def _axes(self, inds):

       return self._inds.axes(*inds)       


   def _output_inds(self, inds):

       return self._inds.remove(*inds)


   def _apply(fun, inds=None, **opts):

       if inds is None:

          data = fun(self._data, **opts)

          return core.TensorGen(data, Indices())

       inds = self._map(inds)
       data = fun(self._data, self._axes(inds), **opts)

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


   def cumsum(self, inds=None, dtype=None, **opts):

       return self._apply(ar.cumsum, inds, dtype, **opts)


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
def cumsum(x, inds=None, dtype=None, **opts):

    op = tensor_reduce(x)
    return op.cumsum(inds, dtype, **opts)




# --- Linear algebra methods ------------------------------------------------ #

@ad.differentiable
def norm(x, inds=None, order=None, **opts):

    op = tensor_reduce(x)
    return op.norm(inds, order, **opts)  









"""


###############################################################################
###                                                                         ###
###  Definitions of non-differentiable tensor operations                    ###
###                                                                         ###
###############################################################################


# --- Basic functionality --------------------------------------------------- #

@ad.nondifferentiable
def copy(x, **opts):

    return x.copy(**opts)


@ad.nondifferentiable
def todense(x):

    return x.todense()


@ad.nondifferentiable
def withdata(x, data):

    return x.withdata(data)


@ad.nondifferentiable
def space(x):

    return x.space()


@ad.nondifferentiable
def item(x, *pos):

    return x.item(*pos)




# --- Tensor properties ----------------------------------------------------- #

@ad.nondifferentiable
def dtype(x):

    return x.dtype


@ad.nondifferentiable
def size(x):

    return x.size


@ad.nondifferentiable
def ndim(x):

    return x.ndim


@ad.nondifferentiable
def shape(x):

    return x.shape




# --- Value methods --------------------------------------------------------- #











###############################################################################
###                                                                         ###
###  Definitions of differentiable tensor operations                        ###
###                                                                         ###
###############################################################################






# --- Shape methods --------------------------------------------------------- #





# --- Value methods --------------------------------------------------------- #

@ad.differentiable
def getitem(x, pos):

    return x[pos]




# --- Standard math --------------------------------------------------------- #

@ad.differentiable
def neg(x):

    return x.neg()


"""
 

"""

class TensorOpUnary(TensorOp):

   def __init__(self, data, inds): 

       self._data = data
       self._inds = inds


   # --- Value methods --- #

   def __getitem__(self, pos):

       return TensorGen(self._data[pos])


   def allof(self, inds=None, **opts):

       if inds is None:
          inds = 

       inds = self._inds.map(inds)
       self._inds.remove()

       
       data = ar.allof(self._data, axis, **opts)

       return TensorGen(data, )
  


   # --- Standard math --- #

   def neg(self):

       return ar.neg(self._data)

"""



