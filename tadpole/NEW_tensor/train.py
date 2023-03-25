#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util as util




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
































