#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.array    as ar
import tadpole.index    as tid

import tadpole.tensor.element         as telem
import tadpole.tensor.core            as core
import tadpole.tensor.elemwise_unary  as unary
import tadpole.tensor.elemwise_binary as binary


from tadpole.tensor.types import (
   Pluggable,
   Tensor, 
   Space,
   Element,
)


from tadpole.index import (
   Index,
   IndexGen,  
   Indices,
)




###############################################################################
###                                                                         ###
###  Tensor creation functions (from indices)                               ###
###                                                                         ###
###############################################################################


# --- Gradient factories ---------------------------------------------------- #

def sparsegrad(inds, elem, vals, dtype=None, backend=None):

    space = tensorspace(inds=inds, dtype=dtype, backend=backend)
    return space.sparsegrad(elem, vals)


def nullgrad(inds, dtype=None, backend=None):

    space = tensorspace(inds=inds, dtype=dtype, backend=backend)
    return space.nullgrad()




# --- Tensor factories ------------------------------------------------------ #

def zeros(inds, dtype=None, backend=None, **opts):

    space = tensorspace(inds=inds, dtype=dtype, backend=backend)
    return space.zeros(**opts)


def ones(inds, dtype=None, backend=None, **opts):

    space = tensorspace(inds=inds, dtype=dtype, backend=backend)
    return space.ones(**opts)


def unit(inds, pos, dtype=None, backend=None, **opts):

    space = tensorspace(inds=inds, dtype=dtype, backend=backend)
    return space.unit(pos, **opts)


def eye(inds, lind=None, rind=None, dtype=None, backend=None, **opts):

    space = tensorspace(inds=inds, dtype=dtype, backend=backend)
    return space.eye(lind=lind, rind=rind, **opts)


def rand(inds, dtype=None, backend=None, **opts):

    space = tensorspace(inds=inds, dtype=dtype, backend=backend)
    return space.rand(**opts)


def randn(inds, dtype=None, backend=None, **opts):

    space = tensorspace(inds=inds, dtype=dtype, backend=backend)
    return space.randn(**opts)


def randuniform(inds, boundaries, dtype=None, backend=None, **opts):

    space = tensorspace(inds=inds, dtype=dtype, backend=backend)
    return space.randuniform(boundaries, **opts)




# --- Tensor generators ----------------------------------------------------- #

def units(inds, dtype=None, backend=None, **opts):

    space = tensorspace(inds=inds, dtype=dtype, backend=backend)
    return space.units(**opts)


def basis(inds, dtype=None, backend=None, **opts):

    space = tensorspace(inds=inds, dtype=dtype, backend=backend)
    return space.basis(**opts)




###############################################################################
###                                                                         ###
###  Tensor space                                                           ###
###                                                                         ###
###############################################################################


# --- TensorSpace factory --------------------------------------------------- #

def tensorspace(inds=None, dtype=None, backend=None, **opts):

    if inds is None:
       inds = Indices()

    if not isinstance(inds, Indices):
       inds = Indices(*inds)

    arrayspace = ar.arrayspace(inds.shape, dtype=dtype, backend=backend)

    return TensorSpace(arrayspace, inds)




# --- TensorSpace ----------------------------------------------------------- #

class TensorSpace(Space):

   # --- Construction --- #

   def __init__(self, arrayspace, inds):

       if arrayspace.shape != inds.shape:
          raise ValueError((
             f"{type(self).__name__}: "
             f"array space and indices must have matching shapes, but array "
             f"space shape {arrayspace.shape} != index shape {inds.shape}"
          ))

       self._arrayspace = arrayspace
       self._inds       = inds


   # --- Comparisons --- #

   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          log.val(self._arrayspace, other._arrayspace)
          log.val(self._inds,       other._inds)

       return bool(log)


   # --- Fill space with data --- #

   def fillwith(self, data):

       return core.astensor(self._arrayspace.fillwith(data), self._inds)


   # --- Reshape space --- #

   def reshape(self, inds):

       arrayspace = self._arrayspace.reshape(tid.shapeof(*inds))

       return self.__class__(arrayspace, inds)


   # --- Gradient factories --- #

   def sparsegrad(self, elem, vals):

       if not isinstance(elem, Element):
          elem = telem.elem(*elem)    
  
       return core.SparseGrad(
                 self, 
                 elem.grid(self._inds), 
                 elem.align(vals, self._inds)
              )


   def nullgrad(self):

       return core.NullGrad(
                 self
              )


   # --- Tensor factories --- #

   def zeros(self, **opts):

       return core.astensor(
                 self._arrayspace.zeros(**opts), 
                 self._inds
              )
 

   def ones(self, **opts):

       return core.astensor(
                 self._arrayspace.ones(**opts), 
                 self._inds
              )


   def unit(self, pos, **opts):

       return core.astensor(
                 self._arrayspace.unit(pos, **opts), 
                 self._inds
              )


   def eye(self, lind=None, rind=None):

       if not lind and not rind:
          lind, rind = self._inds

       lind, rind = self._inds.map(lind, rind)
       data       = self._arrayspace.eye(len(lind), len(rind))

       return core.astensor(data, (lind, rind))


   def rand(self, **opts):

       return core.astensor(
                 self._arrayspace.rand(**opts), 
                 self._inds
              )


   def randn(self, **opts):

       return core.astensor(
                 self._arrayspace.randn(**opts), 
                 self._inds
              )


   def randuniform(self, boundaries, **opts):

       return core.astensor(
                 self._arrayspace.randuniform(boundaries, **opts), 
                 self._inds
              )


   def units(self, **opts):

       for data in self._arrayspace.units(**opts):
           yield core.astensor(data, self._inds)


   def basis(self, **opts):

       for data in self._arrayspace.basis(**opts):
           yield core.astensor(data, self._inds)


   # --- Space properties --- #

   @property
   def dtype(self):
       return self._arrayspace.dtype

   @property
   def size(self):
       return self._arrayspace.size 

   @property 
   def ndim(self):
       return self._arrayspace.ndim

   @property
   def shape(self):
       return self._arrayspace.shape




