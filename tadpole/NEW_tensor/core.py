#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.array    as ar

import tadpole.tensor.unary  as unary
import tadpole.tensor.binary as binary


from tadpole.tensor.types import (
   Tensor, 
   Pluggable,
)


from tadpole.tensor.index import (
   Index, 
   Indices,
   shapeof, 
   sizeof,
)



# --- General tensor -------------------------------------------------------- #

class TensorGen(Tensor, Pluggable):

   # --- Construction --- #

   def __init__(self, data, inds):

       if data.shape != inds.shape,
          raise ValueError((
             f"{type(self).__name__}: 
             f"data and indices must have matching shapes, "
             f"but data shape {data.shape} != index shape {inds.shape}"
          ))

       self._data = data
       self._inds = inds


   # --- Plugging into function calls --- #

   def pluginto(self, op):

       return op.attach(self._data, self._inds)


   # --- Arithmetics and element access --- # 

   def __getitem__(self, pos):

       return unary.getitem(self, pos)


   def __neg__(self):

       return unary.neg(self)


   def __add__(self, other):

       return binary.add(self, other)





# --- Tensor member methods: basic functionality ---------------------------- #

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




# --- Tensor member methods: properties ------------------------------------- #

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











































