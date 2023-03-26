#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util as util

from tadpole.tensor.types import (
   Engine,
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


# --- Elementwise engine ---------------------------------------------------- #

class EngineElemwise(Engine): 

   def __init__(self, optype, size, train=None):

       if train is None:
          train = TrainTensorData()

       self._optype = optype
       self._size   = size
       self._train  = train


   def _inds(self):

       inds = max(self._train.inds(), key=len)

       assert set(util.concat(self._train.inds())) == set(inds), (
          f"\n{type(self).__name__}.operator: "
          f"An elementwise operation cannot be performed for tensors"
          f"with incompatible indices {tuple(self._train.inds())}."
       )

       return inds


   def attach(self, data, inds):

       if self._train.size() == self._size:
          raise TooManyArgsError(self, self._size)

       return self.__class__(
          self._optype, self._size, self._train.attach(data, inds)
       )


   def operator(self):
       
       return self._optype(*self._train.data(), self._inds())




