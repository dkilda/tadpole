#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np

import tadpole.util as util
import tadpole.autodiff as ad

import tadpole.array.operations as op
import tadpole.array.backends   as backends




###############################################################################
###                                                                         ###
###  Array creation functions                                               ###
###                                                                         ###
###############################################################################


# --- Array factories ------------------------------------------------------- #

def fromfun(fun, backend, *datas, **opts):

    return DataFun(fun)(
              backend, *datas, **opts
           )


def asarray(backend, data, **opts):

    if isinstance(data, ArrayLike):
       return data

    return DataFun(lambda backend_, data_: data_)(
              backend, data, **opts
           )


def zeros(backend, shape, **opts):

    return ShapeFun("zeros")(
              backend, shape, **opts
           )


def ones(backend, shape, **opts):

    return ShapeFun("ones")(
              backend, shape, **opts
           )


def unit(backend, shape, idx, **opts):

    return ShapeFun("unit")(
              backend, shape, idx, **opts
           )


def rand(backend, shape, **opts):

    return ShapeFun("rand")(
              backend, shape, **opts
           )


def randn(backend, shape, **opts):

    return ShapeFun("randn")(
              backend, shape, **opts
           )


def randuniform(backend, shape, boundaries, **opts):

    return ShapeFun("randuniform")(
              backend, shape, boundaries, **opts
           )




# --- Array generators ------------------------------------------------------ #

def units(backend, shape, dtype=None):

    for idx in np.ndindex(*shape):
        yield unit(backend, shape, idx, dtype=dtype)



def basis(backend, shape, dtype=None): 

    backend   = backends.get(backend)
    dtype     = backend.get_dtype(dtype)
    gen_units = units(backend.name(), shape, dtype=dtype)

    if  dtype in backend.complex_dtypes():

        for unit in gen_units:
            yield unit
            yield 1j * unit

    else:
        for unit in gen_units:
            yield unit




# --- Data function wrapper ------------------------------------------------- #

class DataFun:

   def __init__(self, fun):

       self._fun = fun


   def __call__(self, backend, *datas, **opts):

       backend = backends.get(backend)                                        
       newdata = self._fun(backend, *datas)
       newdata = backend.asarray(newdata, **opts)

       return Array(backend, newdata)

       


# --- Shape function wrappe-------------------------------------------------- #

class ShapeFun:

   def __init__(self, fun):

       self._fun = fun


   def __call__(self, backend, shape, *args, **opts):

       backend = backends.get(backend)
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

   @abc.abstractmethod
   def apply(self, fun, *datas):
       pass

   @abc.abstractmethod
   def visit(self, fun, *datas):
       pass

   @abc.abstractmethod
   def asarray(self, data):
       pass

   @property
   @abc.abstractmethod
   def dtype(self):
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

   def __init__(self, backend, shape, dtype):

       self._backend = backend
       self._dtype   = dtype
       self._shape   = shape


   def __eq__(self, other):

       log = util.LogicalChain()

       log.typ(self, other)
       log.val(self._backend, other._backend)
       log.val(self._dtype,   other._dtype)
       log.val(self._shape,   other._shape)

       return bool(log)


   def _create(self, fun, *args, **opts):

       return fun(
          self._backend, self._shape, *args, dtype=self._dtype, **opts
       )


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


   def apply(self, fun, *datas):

       return fromfun(fun, self._backend, *datas, dtype=self._dtype)


   def visit(self, fun, *datas):
               
       return fun(backends.get(self._backend), *datas)


   def asarray(self, data):

       return asarray(self._backend, data, dtype=self._dtype)


   @property
   def dtype(self):
       return self._dtype

   @property 
   def ndim(self):
       return len(self._shape)

   @property
   def shape(self):
       return self._shape




###############################################################################
###                                                                         ###
###  Definition of array.                                                   ###
###                                                                         ###
###############################################################################


# --- ArrayLike interface --------------------------------------------------- #

class ArrayLike(abc.ABC):

   @abc.abstractmethod
   def copy(self, **opts):
       pass

   @abc.abstractmethod
   def space(self):
       pass

   @abc.abstractmethod
   def pluginto(self, funcall):
       pass

   @property
   @abc.abstractmethod
   def dtype(self):
       pass

   @property
   @abc.abstractmethod 
   def ndim(self):
       pass

   @property
   @abc.abstractmethod
   def shape(self):
       pass

   @abc.abstractmethod
   def allclose(self, other, **opts):
       pass

   @abc.abstractmethod
   def __eq__(self, other):
       pass

   @abc.abstractmethod
   def __getitem__(self, idx):
       pass

   @abc.abstractmethod
   def __neg__(self):
       pass

   @abc.abstractmethod
   def __add__(self, other):
       pass

   @abc.abstractmethod
   def __mul__(self, other):
       pass

   @abc.abstractmethod
   def __radd__(self, other):
       pass

   @abc.abstractmethod
   def __rmul__(self, other):
       pass




# --- Array ----------------------------------------------------------------- #

class Array(ArrayLike):

   def __init__(self, backend, data):

       self._backend = backend
       self._data    = data


   def copy(self, deep=True):

       data = self._backend.copy(self._data) if deep else self._data

       return self.__class__(self._backend, data)


   def space(self):

       return ArraySpace(self._backend, self.shape, self.dtype)


   def pluginto(self, funcall):

       return funcall.attach(self, self._data)


   @property
   def dtype(self):
       return self._backend.dtype(self._data)

   @property 
   def ndim(self):
       return self._backend.ndim(self._data)

   @property
   def shape(self):
       return self._backend.shape(self._data)


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
          return util.allequal(self._data, other._data)   

       return False


   def __getitem__(self, idx):

       return op.getitem(self, idx)  


   def __neg__(self):

       return op.neg(self)


   def __add__(self, other):

       return op.add(self, other)


   def __mul__(self, other):

       return op.mul(self, other)


   def __radd__(self, other):

       if other == 0:
          return self

       return op.add(self, asarray(self._backend, other))


   def __rmul__(self, other):

       if other == 1:
          return self

       return op.mul(self, asarray(self._backend, other))




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


