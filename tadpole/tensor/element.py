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


# --- Tensor element by indices --------------------------------------------- #

class ElementByIndices(Element):

   def __init__(self, inds, positions):

       self._inds      = inds
       self._positions = positions


   def positions(self, tensor_inds):

       axes = tensor_inds.axes(*tensor_inds.map(*self._inds))

       return tuple(util.relsort(self._positions, axes))
       

   def inds(self, tensor_inds):

       inds = list(tensor_inds)

       for i, ind in enumerate(inds):
           if ind in self._inds:
              inds[i] = ind.resize(len(self._positions[i])) 

       return Indices(*inds)




# --- Tensor element by axes ------------------------------------------------ #

class ElementByAxes(Element):

   def __init__(self, positions):

       self._positions = positions


   def positions(self, tensor_inds):

       return self._positions


   def inds(self, tensor_inds):

       inds = list(tensor_inds)  

       for i, ind in enumerate(inds):
           if i < len(self._positions):
              inds[i] = ind.resized(len(self._positions[i])) 

       return Indices(*inds)




# --- Tensor element factory ------------------------------------------------ #

def _elem_by_inds(pos_by_ind):

    return ElementByIndices(
       tuple(pos_by_ind.keys()), 
       tuple(pos_by_ind.values())
    )


def _elem_by_axes(pos):

    return ElementByAxes(pos)


def elem(*args, **kwargs):

    if len(args) > 0:
       if len(kwargs) > 0:
           raise ValueError(
              f"elem: the input should be provided using "
              f"either args or kwargs, but not both. "
              f"Args: {args}, Kwargs: {kwargs}. "
           )

    if len(args) == 0:
       return _elem_by_inds(kwargs)

    if isinstance(args[0], dict):
       return _elem_by_inds(args[0])

    return _elem_by_axes(*args)




