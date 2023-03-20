#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.array    as ar

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



"""
###############################################################################
###                                                                         ###
###  Tensor creation functions (from indices)                               ###
###                                                                         ###
###############################################################################


# --- Factory that constructs a Tensor from function ------------------------ #

class TensorFromFun:

   def __init__(self, funstr):

       self._funstr = funstr


   @property
   def _fun(self):

      return {
              "zeros":       ar.zeros,
              "ones":        ar.ones,
              "unit":        ar.unit,
              "rand":        ar.rand,
              "randn":       ar.randn,
              "randuniform": ar.randuniform,
             }[self._funstr]


   def __call__(self, inds, *args, **opts):
 
       data = self._fun(inds.shape, *args, **opts)
       return Tensor(data, inds)  




# --- Tensor factories ------------------------------------------------------ #

@ad.differentiable
def sparse(inds, pos, vals, **opts):

    void  = ar.make_void(**opts)
    space = TensorSpace(void, inds, opts.get("dtype", None))

    if "dtype" in opts:
       vals = ar.asarray(vals, **opts)
       vals = ar.astype(vals, dtype=opts["dtype"])

    return SparseGrad(
              space, inds, pos, vals
           )









@from_arrayspace
def zeros(arrayspace, inds, **opts):

    return zeros(arrayspace, inds, **opts)



@ad.nondifferentiable
def zeros(inds, **opts):

    arrayspace = ar.space(inds.shape, **opts)

    return zeros(arrayspace, inds, **opts)

    return TensorFromFun("zeros")(
              inds, **opts
           )


@ad.nondifferentiable
def zeros(arrayspace, inds, **opts):

    data = arrayspace.zeros(**opts)
    return Tensor(data, inds)




@ad.nondifferentiable
def ones(inds, **opts):

    return TensorFromFun("ones")(
              inds, **opts
           )


@ad.nondifferentiable
def unit(inds, pos, **opts):

    return TensorFromFun("unit")(
              inds, pos, **opts
           )


@ad.nondifferentiable
def rand(inds, **opts):

    return TensorFromFun("rand")(
              inds, **opts
           )


@ad.nondifferentiable
def randn(inds, **opts):

    return TensorFromFun("randn")(
              inds, **opts
           )


@ad.nondifferentiable
def randuniform(inds, boundaries, **opts):

    return TensorFromFun("randuniform")(
              inds, boundaries, **opts
           )




# --- Tensor generators ----------------------------------------------------- #

@ad.nondifferentiable
def units(inds, **opts):

    for pos in np.ndindex(*inds.shape):
        yield unit(inds, pos, **opts)




@ad.nondifferentiable
def basis(inds, dtype=None, **opts): 

    if  ar.iscomplex_type(dtype, **opts):

        for unit in units(inds, dtype=dtype, **opts):
            yield unit
            yield 1j * unit

    else:
        for unit in units(inds, dtype=dtype, **opts):
            yield unit

"""










###############################################################################
###                                                                         ###
###  Tensor creation functions (from indices)                               ###
###                                                                         ###
###############################################################################


# --- Tensor factories ------------------------------------------------------ #

@ad.differentiable
def sparse_from_space(arrayspace, inds, pos, vals):

    space = TensorSpace(arrayspace, inds)
    vals  = ar.asarray(vals, dtype=space.dtype)

    return SparseGrad(space, inds, pos, vals)



@ad.nondifferentiable
def zeros_from_space(arrayspace, inds, **opts):

    data = arrayspace.zeros(**opts)
    return Tensor(data, inds)
 


@ad.nondifferentiable
def ones_from_space(arrayspace, inds, **opts):

    data = arrayspace.ones(**opts)
    return Tensor(data, inds)



@ad.nondifferentiable
def unit_from_space(arrayspace, inds, pos, **opts):

    data = arrayspace.unit(pos, **opts)
    return Tensor(data, inds)



@ad.nondifferentiable
def rand_from_space(arrayspace, inds, **opts):

    data = arrayspace.rand(**opts)
    return Tensor(data, inds)



@ad.nondifferentiable
def randn_from_space(arrayspace, inds, **opts):

    data = arrayspace.randn(**opts)
    return Tensor(data, inds)



@ad.nondifferentiable
def randuniform_from_space(arrayspace, inds, boundaries, **opts):

    data = arrayspace.randn(boundaries, **opts)
    return Tensor(data, inds)




# --- Tensor generators ----------------------------------------------------- #

@ad.nondifferentiable
def units_from_space(arrayspace, inds, **opts):

    for data in arrayspace.units(**opts):
        yield Tensor(data, inds)



@ad.nondifferentiable
def basis_from_space(arrayspace, inds, **opts):

    for data in arrayspace.basis(**opts):
        yield Tensor(data, inds)



"""
    for pos in np.ndindex(*inds.shape):
        yield unit(space, inds, pos, **opts)
"""
 

"""
    if  ar.iscomplex_type(dtype, **opts):

        for unit in units(inds, dtype=dtype, **opts):
            yield unit
            yield 1j * unit

    else:
        for unit in units(inds, dtype=dtype, **opts):
            yield unit
"""


# --- Decorator that creates ArraySpace for tensor factories ---------------- #

def auto_arrayspace(fun):

    def wrap(inds, **opts):

        arrayspace = ar.space(inds.shape, **opts)
        return fun(arrayspace, inds, **opts)

    return wrap




# --- Tensor factories with automatic ArraySpace ---------------------------- #

sparse = auto_arrayspace(sparse_from_space)
zeros  = auto_arrayspace(zeros_from_space)
ones   = auto_arrayspace(ones_from_space)
unit   = auto_arrayspace(unit_from_space)

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

   # --- Factories --- #

   @abc.abstractmethod
   def sparse(self, pos, vals):
       pass

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

   def __init__(self, arrayspace, inds):

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


   # --- Factories --- #

   def sparse(self, pos, vals):

       return self._create(
          sparse_from_space, pos, vals
       )


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



 



