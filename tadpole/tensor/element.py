#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.array    as ar
import tadpole.index    as tid


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

def upperbounded(positions, inds):

    def _bounded(pos, ind):

        if isinstance(pos, slice) and pos.stop is None:
           return slice(pos.start, len(ind), pos.step)

        return pos

    return tuple(map(_bounded, positions, inds)) 


def resized(ind, pos):

    if isinstance(pos, slice):
       return ind.resized(pos.indices(pos.stop)[0], pos.stop)

    return ind.resized(pos, pos)


def nonzero_sized(inds):

    return Indices(*(ind for ind in inds if len(ind) > 0)) 


def grid(pos):

    def toslice(x):
        if isinstance(x, slice):
           return x
        return slice(x, x+1)

    pos = tuple(map(toslice, pos))

    return tuple(map(ar.asarray, np.mgrid[pos])) 


def align(vals, pos):

    axes = tuple(i for i, x in enumerate(pos) if not isinstance(x, slice))
    vals = ar.asarray(vals)

    if len(axes) < vals.ndim:
       vals = ar.unsqueeze(vals, axes)

    return vals




# --- Tensor element by indices --------------------------------------------- #

class ElementByIndices(Element):

   def __init__(self, inds, positions):

       self._inds      = inds
       self._positions = positions


   def pos(self, inds):

       pos  = upperbounded(self._positions, inds.map(*self._inds))
       axes = util.argsort(inds.axes(*inds.map(*self._inds)))

       return tuple(pos[axis] for axis in axes)


   def grid(self, inds):

       return grid(self.pos(inds))


   def align(self, vals, inds):

       return align(vals, self.pos(inds))

       
   def inds(self, inds):
       
       pos        = upperbounded(self._positions, inds.map(*self._inds))
       pos_by_ind = dict(zip(inds.map(*self._inds), pos))

       inds = list(inds)
       for i, ind in enumerate(inds):
           if ind in pos_by_ind:
              inds[i] = resized(ind, pos_by_ind[ind])

       return nonzero_sized(inds) 




# --- Tensor element by axes ------------------------------------------------ #

class ElementByAxes(Element):

   def __init__(self, positions):

       self._positions = positions


   def pos(self, inds):

       return upperbounded(self._positions, inds)


   def grid(self, inds):

       return grid(self.pos(inds))


   def align(self, vals, inds):

       return align(vals, self.pos(inds))


   def inds(self, inds):

       inds = list(inds) 
       pos  = self.pos(inds) 

       for i, ind in enumerate(inds):

           if i == len(pos):
              break

           inds[i] = resized(ind, pos[i]) 

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




