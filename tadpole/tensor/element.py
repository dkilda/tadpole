#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.array    as ar
import tadpole.index    as tid

import tadpole.tensor.space as sp


from tadpole.tensor.types import (
   Element,
)


from tadpole.index import (
   Index,
   IndexGen, 
   Indices,
)




###############################################################################
###                                                                         ###
###  Tensor elements                                                        ###
###                                                                         ###
###############################################################################


# --- Helpers for index manipulation ---------------------------------------- #

def resized(ind, pos):

    if isinstance(pos, slice):
       return ind.resized(pos.indices(pos.stop)[0], pos.stop)

    return ind.resized(pos, pos)


def nonzero_sized(inds):

    return Indices(*(ind for ind in inds if len(ind) > 0)) 




# --- Tensor element by indices --------------------------------------------- #

class ElementByIndices(Element):

   def __init__(self, inds, positions):

       self._inds      = inds
       self._positions = positions


   def pos(self, inds):

       axes = util.argsort(inds.axes(*inds.map(*self._inds)))

       return tuple(self._positions[axis] for axis in axes)
       

   def inds(self, inds):

       pos_by_ind = dict(zip(inds.map(*self._inds), self._positions))
       inds       = list(inds)

       for i, ind in enumerate(inds):
           if ind in pos_by_ind:
              inds[i] = resized(ind, pos_by_ind[ind])

       return nonzero_sized(inds) 




# --- Tensor element by axes ------------------------------------------------ #

class ElementByAxes(Element):

   def __init__(self, positions):

       self._positions = positions


   def pos(self, inds):

       return self._positions


   def inds(self, inds):

       inds = list(inds)  

       for i, ind in enumerate(inds):

           if i == len(self._positions):
              break

           inds[i] = resized(ind, self._positions[i]) 

       return nonzero_sized(inds) 




# --- Tensor element factory ------------------------------------------------ #

def elem(*args, **kwargs):

    if len(args) > 0:
       if len(kwargs) > 0:
           raise ValueError(
              f"elem: the input should be provided using "
              f"either args or kwargs, but not both. "
              f"Args: {args}, Kwargs: {kwargs}. "
           )

    if len(args) == 0:
       return ElementByIndices(*zip(*kwargs.items()))

    if isinstance(args[0], dict):
       return ElementByIndices(*zip(*args[0].items()))

    return ElementByAxes(args)




