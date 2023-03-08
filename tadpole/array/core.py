#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np

import tadpole.util     as util
import tadpole.autodiff as ad

import tadpole.array.operations as op
import tadpole.array.backends   as backends
import tadpole.array.grad       as grad

from tadpole.array.types import (
   Space,
   Pluggable,
   ArrayLike,
)




###############################################################################
###                                                                         ###
###  Array creation functions                                               ###
###                                                                         ###
###############################################################################


# --- Array generators ------------------------------------------------------ #

def units(shape, dtype=None, backend=None):

    for idx in np.ndindex(*shape):
        yield unit(shape, idx, dtype=dtype, backend=backend)



def basis(shape, dtype=None, backend=None): 

    backend   = backends.get(backend)
    dtype     = backend.get_dtype(dtype)
    gen_units = units(shape, dtype=dtype, backend=backend)

    if  dtype in backend.complex_dtypes():

        for unit in gen_units:
            yield unit
            yield 1j * unit

    else:
        for unit in gen_units:
            yield unit




# --- Array factories (from data) ------------------------------------------- #

def fromfun(fun, *datas, **opts):

    backend = backends.get_from(opts)
                                        
    outdata = fun(backend, *datas)
    outdata = backend.asarray(outdata, **opts)

    return Array(backend, outdata)


def asarray(data, **opts):

    if isinstance(data, ArrayLike):
       return data

    backend = backends.get_from(opts)                            
    data    = backend.asarray(data, **opts)

    return Array(backend, data)




# --- Array factories (from shape) ------------------------------------------ #

def sparse(shape, idxs, vals, **opts):

    backend = backends.get_from(opts)

    if "dtype" in opts:
       vals = backend.asarray(vals)
       vals = backend.astype(vals, **opts)

    return grad.SparseGrad(
              backend, shape, idxs, vals
           )


def zeros(shape, **opts):

    return ArrayFromShape("zeros")(
              shape, **opts
           )


def ones(shape, **opts):

    return ArrayFromShape("ones")(
              shape, **opts
           )


def unit(shape, idx, **opts):

    return ArrayFromShape("unit")(
              shape, idx, **opts
           )


def rand(shape, **opts):

    return ArrayFromShape("rand")(
              shape, **opts
           )


def randn(shape, **opts):

    return ArrayFromShape("randn")(
              shape, **opts
           )


def randuniform(shape, boundaries, **opts):

    return ArrayFromShape("randuniform")(
              shape, boundaries, **opts
           )




# --- Generic factory that constructs an Array from shape input ------------- #

class ArrayFromShape:

   def __init__(self, fun):

       self._fun = fun


   def __call__(self, shape, *args, **opts):

       backend = backends.get_from(opts) 
       fun     = self._fun

       if callable(fun): 
          fun = fun(backend)

       if isinstance(fun, str):
          fun = {
                 "zeros":       backend.zeros,
                 "ones":        backend.ones,
                 "unit":        backend.unit,
                 "rand":        backend.rand,
                 "randn":       backend.randn,
                 "randuniform": backend.randuniform,
                }[fun]

       data = fun(shape, *args, **opts)
       return Array(backend, data)  




###############################################################################
###                                                                         ###
###  Array space                                                            ###
###                                                                         ###
###############################################################################


# --- Space interface ------------------------------------------------------- #

class Space(abc.ABC):

   # --- Factories --- #

   @abc.abstractmethod
   def sparse(self, idxs, vals):
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

       


# --- ArraySpace ------------------------------------------------------------ #

class ArraySpace(Space):

   # --- Construction --- #

   def __init__(self, backend, shape, dtype):

       self._backend = backend
       self._dtype   = dtype
       self._shape   = shape


   # --- Comparisons --- #

   def __eq__(self, other):

       log = util.LogicalChain()

       log.typ(self, other)
       log.val(self._backend, other._backend)
       log.val(self._dtype,   other._dtype)
       log.val(self._shape,   other._shape)

       return bool(log)


   # --- Factories --- #

   def sparse(self, idxs, vals):

       return self._create(sparse, idxs, vals)


   def zeros(self):

       return self._create(zeros) 


   def ones(self):

       return self._create(ones) 


   def unit(self):

       return self._create(unit) 


   def rand(self, **opts):

       return self._create(rand, **opts) 


   def randn(self, **opts):

       return self._create(randn, **opts) 


   def randuniform(self, boundaries, **opts):

       return self._create(randuniform, boundaries, **opts) 


   def units(self):

       return self._create(units) 


   def basis(self):

       return self._create(basis) 


   # --- Space properties --- #

   @property
   def dtype(self):
       return self._dtype

   @property
   def size(self):
       return np.prod(self._shape)

   @property 
   def ndim(self):
       return len(self._shape)

   @property
   def shape(self):
       return self._shape


   # --- Private helpers --- #

   def _create(self, fun, *args, **opts):

       return fun(
          self._shape, *args, 
          dtype=self._dtype, backend=self._backend, **opts
       )




###############################################################################
###                                                                         ###
###  Definition of array.                                                   ###
###                                                                         ###
###############################################################################


# --- Array ----------------------------------------------------------------- #

class Array(ArrayLike, Pluggable):

   # --- Construction --- #

   def __init__(self, backend, data):

       if not isinstance(backend, backends.backend.Backend):
          raise ValueError(
             f"Array: backend must be an instance "
             f"of Backend, but it is {backend}"
          )  

       self._backend = backend
       self._data    = data


   # --- Plugging into function calls --- #

   def pluginto(self, funcall):

       return funcall.attach(self._backend, self._data)


   # --- Using in gradient accumulations --- #

   def addto(self, other):

       if isinstance(other, grad.ZeroGrad): 
          return self

       if isinstance(other, grad.SparseGrad):
          return other.addto(self)

       data = self._backend.add(self._data, other._data)
       return self.asarray(data)


   # --- Basic functionality --- #

   def copy(self, deep=True):

       data = self._backend.copy(self._data) if deep else self._data

       return self.__class__(self._backend, data)


   def asarray(self, data):

       return asarray(data, backend=self._backend)


   def space(self):

       return ArraySpace(self._backend, self.shape, self.dtype)


   def item(self, *idx): 

       return self._backend.item(self._data, *idx)


   # --- Array properties --- #

   @property
   def dtype(self):
       return self._backend.dtype(self._data)

   @property 
   def size(self):
       return self._backend.size(self._data)  

   @property 
   def ndim(self):
       return self._backend.ndim(self._data)  

   @property
   def shape(self):
       return self._backend.shape(self._data)


   # --- Comparisons --- #

   def allclose(self, other, **opts):

       log = util.LogicalChain()
       log.typ(self, other)
       log.val(self._backend, other._backend)

       if bool(log):
          return util.allclose(self._data, other._data, **opts)   

       return False


   def __eq__(self, other):
 
       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          log.val(self._backend, other._backend)

       if bool(log):
          return util.allclose(self._data, other._data)  
          #return util.allequal(self._data, other._data) # FIXME decide what to do with Array equality!  

       return False


   # --- Arithmetics and element access --- # 

   def __getitem__(self, idx):

       return op.getitem(self, idx)  


   def __neg__(self):

       return op.neg(self)


   def __add__(self, other):

       return op.add(self, other)


   def __sub__(self, other):

       return op.sub(self, other)


   def __mul__(self, other):

       return op.mul(self, other)


   def __radd__(self, other):

       return self.__add__(other)

 
   def __rsub__(self, other):

       return -self.__sub__(other)


   def __rmul__(self, other):

       return self.__mul__(other)




###############################################################################
###                                                                         ###
###  Comparison methods for arrays.                                         ###
###                                                                         ###
###############################################################################


# --- Exact equality of arrays ---------------------------------------------- #

def allequal(x, y):

    return x.allequal(y)




# --- Approximate equality of arrays ---------------------------------------- #

def allclose(x, y, **opts):

    return x.allclose(y, **opts)




# --- Exact equality of iterables of arrays --------------------------------- #

def allallequal(xs, ys):

    return all(allequal(x, y) for x, y in zip(xs, ys))




# --- Approximate equality of iterables of arrays --------------------------- #

def allallclose(xs, ys, **opts):

    return all(allclose(x, y, **opts) for x, y in zip(xs, ys))




# TODO we need a broadcasting method!






