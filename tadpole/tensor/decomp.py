#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.array    as ar
import tadpole.index    as tid

import tadpole.tensor.core       as core
import tadpole.tensor.truncation as truncation


from tadpole.tensor.types import (
   Engine,
   CutoffMode,
   ErrorMode,
   Trunc,
   Alignment,
)


from tadpole.tensor.engine import (
   TrainTensorData,
   TooManyArgsError,
)


from tadpole.tensor.truncation import (
   TruncNull,
)


from tadpole.index import (
   Index,
   IndexGen, 
   Indices,
)




###############################################################################
###                                                                         ###
###  The logic of index partitioning: defines the alignment by left/right   ### 
###  indices and a link between the left/right partitions.                  ###
###                                                                         ###
###############################################################################


# --- Left alignment -------------------------------------------------------- #

class LeftAlignment(Alignment):

   def __init__(self, partial_inds):

       self._partial_inds = partial_inds


   def _mapto(self, inds):

       return inds.map(*self._partial_inds)


   def linds(self, inds):

       return Indices(*self._mapto(inds))
       

   def rinds(self, inds):

       return Indices(*inds.remove(*self._mapto(inds))) 




# --- Right alignment -------------------------------------------------------- #

class RightAlignment(Alignment):

   def __init__(self, partial_inds):

       self._partial_inds = partial_inds


   def _mapto(self, inds):

       return inds.map(*self._partial_inds)


   def linds(self, inds):

       return Indices(*inds.remove(*self._mapto(inds))) 
       

   def rinds(self, inds):

       return Indices(*self._mapto(inds))




# --- Link between partitions ----------------------------------------------- #

class Link:

   def __init__(self, name):

       self._name = name
       self._ind  = None


   def ind(self, size):

       if self._ind is None:
          self._ind = IndexGen(self._name, size)

       if size != len(self._ind):
          raise ValueError(
             f"{type(self).__name__}: an attempt to resize link index to "
             f"an incompatible size {size} != original size {len(self._ind)}"
          )

       return self._ind
 



# --- Partition ------------------------------------------------------------- #

class Partition:

   def __init__(self, inds, linds, rinds, link):

       self._inds  = inds
       self._linds = linds
       self._rinds = rinds
       self._link  = link

   
   @property
   def _laxes(self):

       return self._inds.axes(*self._linds)


   @property
   def _raxes(self):

       return self._inds.axes(*self._rinds)


   def aligndata(self, data):

       data = ar.transpose(data, (*self._laxes, *self._raxes))
       data = ar.reshape(data,   (self._linds.size, self._rinds.size))

       return data


   def ltensor(self, data):

       data = ar.reshape(data, (*self._linds.shape, -1))

       sind = self._link.ind(data.shape[-1])
       inds = self._linds.push(sind)

       return core.TensorGen(data, inds)


   def rtensor(self, data):

       data = ar.reshape(data, (-1, *self._rinds.shape))

       sind = self._link.ind(data.shape[0])
       inds = self._rinds.add(sind)

       return core.TensorGen(data, inds)


   def stensor(self, data):

       sind = self._link.ind(data.shape[0])

       return core.TensorGen(data, Indices(sind))




###############################################################################
###                                                                         ###
###  Tensor decomposition engine and operator                               ###
###                                                                         ###
###############################################################################


# --- Tensor decomposition factory ------------------------------------------ #

def tensor_decomp(x, inds, alignment, link):

    link      = Link(link)
    alignment = {
                 "left":  LeftAlignment,
                 "right": RightAlignment,
                }[alignment](inds)

    engine = EngineDecomp(alignment, link)
    engine = x.pluginto(engine)

    return engine.operator()




# --- Tensor decomposition engine ------------------------------------------- #

class EngineDecomp(Engine): 

   def __init__(self, alignment, link, train=None):

       if train is None:
          train = TrainTensorData()

       self._alignment = alignment
       self._link      = link
       self._train     = train


   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          log.val(self._alignment, other._alignment)
          log.val(self._link,      other._link)
          log.val(self._train,     other._train)

       return bool(log)


   @property
   def _size(self):

       return 1


   def attach(self, data, inds):

       if self._train.size() == self._size:
          raise TooManyArgsError(self, self._size)

       return self.__class__(
          self._alignment, 
          self._link, 
          self._train.attach(data, inds)
       )


   def operator(self):

       data, = self._train.data()
       inds, = self._train.inds()

       partition = Partition(
          inds, 
          self._alignment.linds(inds), 
          self._alignment.rinds(inds), 
          self._link
       )

       return TensorDecomp(data, partition)




# --- Tensor decomposition operator ----------------------------------------- #

class TensorDecomp:

   # --- Construction --- #

   def __init__(self, data, partition):

       self._data      = data
       self._partition = partition


   # --- Private helpers --- #

   def _aligned_data(self):

       return self._partition.aligndata(self._data)

  
   def _ltensor(self, data):

       return self._partition.ltensor(data)


   def _stensor(self, data):

       return self._partition.stensor(data)


   def _rtensor(self, data):

       return self._partition.rtensor(data)


   def _explicit(self, fun, trunc):

       output_data = fun(self._aligned_data())
       error       = trunc.error(output_data[1])
       output_data = trunc.apply(*output_data)

       return (
               self._ltensor(output_data[0]), 
               self._stensor(output_data[1]), 
               self._rtensor(output_data[2]), 
               error,
              )

       
   def _hidden(self, fun):

       output_data = fun(self._aligned_data())

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
def svd(x, inds, alignment="left", link="link", trunc=TruncNull()):

    op = tensor_decomp(x, inds, alignment, link)

    return op.svd(trunc)



@ad.differentiable
def eig(x, inds, alignment="left", link="link", trunc=TruncNull()):

    op = tensor_decomp(x, inds, alignment, link)

    return op.eig(trunc)



@ad.differentiable
def eigh(x, inds, alignment="left", link="link", trunc=TruncNull()):

    op = tensor_decomp(x, inds, alignment, link)

    return op.eigh(trunc)




# --- Hidden-rank decompositions -------------------------------------------- #

@ad.differentiable
def qr(x, inds, alignment="left", link="link"):

    op = tensor_decomp(x, inds, alignment, link)

    return op.qr()



@ad.differentiable
def lq(x, inds, alignment="left", link="link"):

    op = tensor_decomp(x, inds, alignment, link)

    return op.lq()




