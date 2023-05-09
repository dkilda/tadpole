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


def nonzero(inds):

    return Indices(*(ind for ind in inds if len(ind) > 0)) 




# --- Tensor element by indices --------------------------------------------- #

class ElementByIndices(Element):

   def __init__(self, pos_by_ind):

       self._pos_by_ind = pos_by_ind


   def pos(self, inds):

       elem_inds = tuple(self._pos_by_ind.keys())
       elem_pos  = tuple(self._pos_by_ind.values())

       return tuple(util.relsort(elem_pos, inds.axes(*inds.map(*elem_inds))))
       

   def inds(self, inds):

       elem_inds = inds.map(*self._pos_by_ind.keys())

       inds = list(inds)

       for i, ind in enumerate(inds):
           if ind in elem_inds:
              inds[i] = resized(ind, self._pos_by_ind[ind])

       return nonzero(inds) 




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

       return nonzero(inds) 




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
       return ElementByIndices(kwargs)

    if isinstance(args[0], dict):
       return ElementByIndices(args[0])

    return ElementByAxes(args)




