#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.array    as ar

import tadpole.tensor.core    as core
import tadpole.tensor.funcall as fn


from tadpole.tensor.types import (
   TensorLike, 
   Pluggable,
)


from tadpole.tensor.index import (
   Index, 
   Indices,
   shapeof, 
   sizeof,
)




###############################################################################
###                                                                         ###
###  Tensor creation functions (from indices)                               ###
###                                                                         ###
###############################################################################


# --- Gradient factories ---------------------------------------------------- #

@ad.differentiable
def sparsegrad_from_space(arrayspace, inds, pos, vals):

    space = TensorSpace(arrayspace, inds)
    vals  = ar.asarray(vals, dtype=space.dtype)

    return core.SparseGrad(space, pos, vals)



@ad.nondifferentiable
def nullgrad_from_space(arrayspace, inds):

    space = TensorSpace(arrayspace, inds)
    return core.NullGrad(space)




# --- Tensor factories ------------------------------------------------------ #

@ad.nondifferentiable
def zeros_from_space(arrayspace, inds, **opts):

    data = arrayspace.zeros(**opts)
    return core.astensor(data, inds)
 


@ad.nondifferentiable
def ones_from_space(arrayspace, inds, **opts):

    data = arrayspace.ones(**opts)
    return core.astensor(data, inds)



@ad.nondifferentiable
def unit_from_space(arrayspace, inds, pos, **opts):

    data = arrayspace.unit(pos, **opts)
    return core.astensor(data, inds)



@ad.nondifferentiable
def rand_from_space(arrayspace, inds, **opts):

    data = arrayspace.rand(**opts)
    return core.astensor(data, inds)



@ad.nondifferentiable
def randn_from_space(arrayspace, inds, **opts):

    data = arrayspace.randn(**opts)
    return core.astensor(data, inds)



@ad.nondifferentiable
def randuniform_from_space(arrayspace, inds, boundaries, **opts):

    data = arrayspace.randn(boundaries, **opts)
    return core.astensor(data, inds)




# --- Tensor generators ----------------------------------------------------- #

@ad.nondifferentiable
def units_from_space(arrayspace, inds, **opts):

    for data in arrayspace.units(**opts):
        yield core.astensor(data, inds)



@ad.nondifferentiable
def basis_from_space(arrayspace, inds, **opts):

    for data in arrayspace.basis(**opts):
        yield core.astensor(data, inds)




# --- Automatic creation of ArraySpace for tensor factories ----------------- #

def auto_arrayspace(fun):

    def wrap(inds, **opts):

        arrayspace = ar.space(inds.shape, **opts)
        return fun(arrayspace, inds, **opts)

    return wrap




# --- Gradient factories with automatic ArraySpace -------------------------- #

sparsegrad = auto_arrayspace(sparsegrad_from_space)
nullgrad   = auto_arrayspace(nullgrad_from_space)




# --- Tensor factories with automatic ArraySpace ---------------------------- #

zeros = auto_arrayspace(zeros_from_space)
ones  = auto_arrayspace(ones_from_space)
unit  = auto_arrayspace(unit_from_space)

rand        = auto_arrayspace(rand_from_space)
randn       = auto_arrayspace(randn_from_space)
randuniform = auto_arrayspace(randuniform_from_space)

units = auto_arrayspace(units_from_space)
basis = auto_arrayspace(basis_from_space)




###############################################################################
###                                                                         ###
###  Tensor space                                                           ###
###                                                                         ###
###############################################################################


# --- Space interface ------------------------------------------------------- #

class Space(abc.ABC):

   # --- Fill the space with data --- #

   @abc.abstractmethod
   def fillwith(self, data):
       pass


   # --- Space indices --- #

   @abc.abstractmethod
   def inds(self, *tags):
       pass

   @abc.abstractmethod
   def __and__(self, other):
       pass

   @abc.abstractmethod
   def __or__(self, other):
       pass

   @abc.abstractmethod
   def __xor__(self, other):
       pass


   # --- Gradient factories --- #

   @abc.abstractmethod
   def sparsegrad(self, pos, vals):
       pass

   @abc.abstractmethod
   def nullgrad(self):
       pass


   # --- Tensor factories --- #

   @abc.abstractmethod
   def zeros(self):
       pass

   @abc.abstractmethod
   def ones(self):
       pass

   @abc.abstractmethod
   def unit(self):
       pass

   @abc.abstractmethod
   def rand(self, **opts):
       pass

   @abc.abstractmethod
   def randn(self, **opts):
       pass

   @abc.abstractmethod
   def randuniform(self, boundaries, **opts):
       pass

   @abc.abstractmethod
   def units(self):
       pass

   @abc.abstractmethod
   def basis(self):
       pass


   # --- Space properties --- #

   @property
   @abc.abstractmethod
   def dtype(self):
       pass

   @property
   @abc.abstractmethod
   def size(self):
       pass

   @property 
   @abc.abstractmethod
   def ndim(self):
       pass

   @property
   @abc.abstractmethod
   def shape(self):
       pass

       


# --- TensorSpace ----------------------------------------------------------- #

class TensorSpace(Space):

   # --- Construction --- #

   def __init__(self, arrayspace, inds=None):

       if inds is None:
          inds = Indices()

       if not isinstance(inds, Indices):
          inds = Indices(*inds)

       self._arrayspace = arrayspace
       self._inds       = inds


   # --- Private helpers --- #

   def _create(self, fun, *args, **opts):

       return fun(self._arrayspace, self._inds, *args, **opts)


   # --- Comparisons --- #

   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          log.val(self._arrayspace, other._arrayspace)
          log.val(self._inds,       other._inds)

       return bool(log)


   # --- Fill the space with data --- #

   def fillwith(self, data):

       return core.astensor(data, self._inds)


   # --- Space indices --- #

   def inds(self, *tags):

       return self._inds.map(*tags)


   def __and__(self, other):

       return self._inds & other._inds


   def __or__(self, other):

       return self._inds | other._inds


   def __xor__(self, other):

       return self._inds ^ other._inds


   # --- Gradient factories --- #

   def sparsegrad(self, pos, vals):

       return self._create(
          sparsegrad_from_space, pos, vals
       )


   def nullgrad(self):

       return self._create(
          nullgrad_from_space
       )


   # --- Tensor factories --- #

   def zeros(self):

       return self._create(
          zeros_from_space
       ) 


   def ones(self):

       return self._create(
          ones_from_space
       ) 


   def unit(self, pos, **opts):

       return self._create(
          unit_from_space, pos, **opts
       ) 


   def rand(self, **opts):

       return self._create(
          rand_from_space, **opts
       ) 


   def randn(self, **opts):

       return self._create(
          randn_from_space, **opts
       ) 


   def randuniform(self, boundaries, **opts):

       return self._create(
          randuniform_from_space, boundaries, **opts
       ) 


   def units(self, **opts):

       return self._create(
          units_from_space, **opts
       )


   def basis(self, **opts):

       return self._create(
          basis_from_space, **opts
       )


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





 

