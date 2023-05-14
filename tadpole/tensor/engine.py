#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import tadpole.util as util


from tadpole.tensor.types import (
   Engine,
)


from tadpole.index import (
   Index,
   IndexGen,  
   Indices,
)





###############################################################################
###                                                                         ###
###  Train of tensor data/metadata                                          ###
###                                                                         ###
###############################################################################


# --- Train of tensor data/metadata ----------------------------------------- #

class TrainTensorData:

   def __init__(self, data=None, inds=None):

       if data is None: data = util.Sequence()
       if inds is None: inds = util.Sequence()

       self._data = data
       self._inds = inds


   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          log.val(self._inds, other._inds)

       if bool(log):
          log.val(self._data, other._data)

       return bool(log)


   def attach(self, data, inds):

       return self.__class__(
          self._data.push(data), self._inds.push(inds)
       ) 


   def size(self):

       return len(self._data)


   def data(self):

       return iter(self._data)


   def inds(self):

       return iter(self._inds)




# --- Max arguments error --------------------------------------------------- #

class TooManyArgsError(Exception):

   def __init__(self, obj, size):

       self._obj  = obj
       self._size = size


   def __str__(self):

       return (
          f"{type(self._obj).__name__}.attach(): "
          f"cannot attach another argument, because  "
          f"{type(self._obj).__name__} already holds the maximum "
          f"allowed number of arguments {self._size}."
       )




###############################################################################
###                                                                         ###
###  Engine for unary operations                                            ###
###                                                                         ###
###############################################################################


# --- Unary engine ---------------------------------------------------------- #

class EngineUnary(Engine):

   def __init__(self, optype, train=None):

       if train is None:
          train = TrainTensorData()

       self._optype = optype
       self._train  = train


   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          log.val(self._optype, other._optype)
          log.val(self._train,  other._train)

       return bool(log)


   @property
   def _size(self):
       
       return 1


   def attach(self, data, inds):

       if self._train.size() == self._size:
          raise TooManyArgsError(self, self._size)

       return self.__class__(
          self._optype, self._train.attach(data, inds)
       )


   def operator(self):

       data, = self._train.data()
       inds, = self._train.inds()

       return self._optype(data, inds)




###############################################################################
###                                                                         ###
###  Engine for elementwise operations                                      ###
###                                                                         ###
###############################################################################


# --- Aligned output index from lined-up input indices ---------------------- # 

def aligned_ind(inds):

    inds               = [ind for ind in inds if ind is not None] 
    non_singleton_inds = [ind for ind in inds if len(ind) > 1] 
    
    if len(non_singleton_inds) == 0:
       return inds[0]

    if len(set(non_singleton_inds)) == 1:
       return non_singleton_inds[0]

    raise ValueError(
       f"aligned_ind: an elementwise operation cannot be "
       f"performed for incompatible indices {inds}."
    )




# --- Elementwise engine ---------------------------------------------------- #

class EngineElemwise(Engine): 

   def __init__(self, optype, size, train=None):

       if train is None:
          train = TrainTensorData()

       self._optype = optype
       self._size   = size
       self._train  = train


   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          log.val(self._optype, other._optype)
          log.val(self._size,   other._size)
          log.val(self._train,  other._train)

       return bool(log)


   def _inds(self):
 
       inds = (reversed(each_inds) for each_inds in self._train.inds())
       inds = reversed(list(itertools.zip_longest(*inds)))

       return Indices(*map(aligned_ind, inds))

       
   def attach(self, data, inds):

       if self._train.size() == self._size:
          raise TooManyArgsError(self, self._size)

       return self.__class__(
          self._optype, self._size, self._train.attach(data, inds)
       )


   def operator(self):
       
       return self._optype(*self._train.data(), self._inds())




