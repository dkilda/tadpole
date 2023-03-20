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




###############################################################################
###                                                                         ###
###  Tensor creation functions                                              ###
###                                                                         ###
###############################################################################


# --- Generic factory that constructs a Tensor from index input ------------- #

class TensorFromInds:

   def __init__(self, fun):

       self._fun = fun


   def __call__(self, inds, *args, **opts):
 
       fun = self._fun

       if isinstance(fun, str):
          fun = {
                 "zeros":       ar.zeros,
                 "ones":        ar.ones,
                 "unit":        ar.unit,
                 "rand":        ar.rand,
                 "randn":       ar.randn,
                 "randuniform": ar.randuniform,
                }[fun]

       data = fun(inds.shape, *args, **opts)
       return Tensor(data, inds)  





def tensor_from_fun(fun):

    def wrap(self, *args, **opts):

        data = fun(self, *args, **opts)
        return core.Tensor(data, self._inds)

    return wrap





# --- Tensor factories (from shape) ------------------------------------------ #

@ad.differentiable
def sparse(inds, pos, vals, **opts):

    if "dtype" in opts:
       vals = ar.asarray(vals)
       vals = ar.astype(vals, **opts)

    return SparseGrad(
              inds, pos, vals, **opts
           )


@ad.nondifferentiable
def zeros(inds, **opts):

    return TensorFromInds("zeros")(
              inds, **opts
           )


@ad.nondifferentiable
def ones(inds, **opts):

    return TensorFromInds("ones")(
              inds, **opts
           )


@ad.nondifferentiable
def unit(inds, pos, **opts):

    return TensorFromInds("unit")(
              inds, pos, **opts
           )


@ad.nondifferentiable
def rand(inds, **opts):

    return TensorFromInds("rand")(
              inds, **opts
           )


@ad.nondifferentiable
def randn(inds, **opts):

    return TensorFromInds("randn")(
              inds, **opts
           )


@ad.nondifferentiable
def randuniform(inds, boundaries, **opts):

    return TensorFromInds("randuniform")(
              inds, boundaries, **opts
           )








# --- Tensor generators ----------------------------------------------------- #

@ad.nondifferentiable
def units(inds, dtype=None, **opts):

    for pos in np.ndindex(*shape):
        yield unit(inds, pos, dtype=dtype, **opts)




@ad.nondifferentiable
def basis(inds, dtype=None, **opts): 

    gen_units = units(inds, dtype=dtype, **opts)

    if  ar.iscomplex_type(dtype):

        for unit in gen_units:
            yield unit
            yield 1j * unit

    else:
        for unit in gen_units:
            yield unit







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

   def __init__(self, void, inds, dtype):

       self._void  = void
       self._inds  = inds
       self._dtype = dtype


   # --- Comparisons --- #

   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          log.val(self._void,  other._void)
          log.val(self._inds,  other._inds)
          log.val(self._dtype, other._dtype)

       return bool(log)


   # --- Factories --- #

   def sparse(self, pos, vals):

       return self._create(sparse, pos, vals)


   def zeros(self):

       return self._create("zeros") 


   def ones(self):

       return self._create("ones") 


   def unit(self, pos, **opts):

       return self._create("unit", pos, **opts) 


   def rand(self, **opts):

       return self._create("rand", **opts) 


   def randn(self, **opts):

       return self._create("randn", **opts) 


   def randuniform(self, boundaries, **opts):

       return self._create("randuniform", boundaries, **opts) 


   def units(self, **opts):

       for pos in np.ndindex(*self.shape):
           yield self.unit(pos, **opts)


   def basis(self, **opts):

       gen_units = self.units(**opts)

       if  self._void.iscomplex_type(dtype):

           for unit in gen_units:
               yield unit
               yield 1j * unit

       else:
           for unit in gen_units:
               yield unit



   # --- Space properties --- #

   @property
   def dtype(self):
       return self._dtype

   @property
   def size(self):
       return self._inds.size 

   @property 
   def ndim(self):
       return self._inds.ndim

   @property
   def shape(self):
       return self._inds.shape


   # --- Private helpers --- #

   def _create(self, fun, *args, **opts):

       if isinstance(fun, str):
          fun = {
                 "zeros":       self._void.zeros,
                 "ones":        self._void.ones,
                 "unit":        self._void.unit,
                 "rand":        self._void.rand,
                 "randn":       self._void.randn,
                 "randuniform": self._void.randuniform,
                }[fun]

       data = fun(self.shape, *args, dtype=self.dtype, **opts)
       return Tensor(data, self._inds)  



