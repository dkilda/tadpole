#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np

import tadpole.array.array_ops as ops
import tadpole.array.backends  as backends
import tadpole.array.util      as util



"""
Up next:

V --- Add .basis() method to ArraySpace

V --- implement the ListRef for a quasi-immutable List-like structure
      (this should be a more versatile alternative to Sequence):

      https://stackoverflow.com/questions/24524409/out-of-place-transformations-on-python-list

V --- sort out the backend module/subpackage

--- implement dense/sparse grads (that will follow ArrayLike interface)

--- make Array implement NodeLike interface: integrate Array with tadpole/autodiff

--- implement all the specific Array operations

--- write tests

"""




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

    for idx in np.index(*shape):
        yield unit(backend, shape, idx, dtype=dtype)



def basis(backend, shape, dtype=None): 

    dtype = backend.get_dtype(dtype)

    if  dtype not in backend.complex_dtypes():
        return units(backend, shape, dtype=dtype)

    for unit in units(backend, shape, dtype=dtype):
        yield unit
        yield 1j * unit




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

       fun = self._fun

       if isinstance(fun, callable):
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

   def __init__(self, backend, dtype, shape):

       self._backend = backend
       self._dtype   = dtype
       self._shape   = shape


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
###  Array and a general framework for array operations                     ###
###                                                                         ###
###############################################################################


# --- ArrayLike interface --------------------------------------------------- #

class ArrayLike(abc.ABC):

   @abc.abstractmethod
   def copy(self):
       pass

   @abc.abstractmethod
   def space(self):
       pass

   @abc.abstractmethod
   def pluginto(self, funcall):
       pass

   @abc.abstractmethod
   def __getitem__(self, coords):
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

class Array(ArrayLike): # TODO update GradAccum/Sum so that get()/pop()/reduce()/etc use 0 and not None as their default!

   def __init__(self, backend, data):

       self._backend = backend
       self._data    = data


   def copy(self):

       return self.__class__(self._backend, self._data)


   def space(self):

       return ArraySpace(self._backend, self.dtype, self.shape)


   def pluginto(self, funcall):

       return funcall.attach(self, self._data)


   def __getitem__(self, coords):

       return self._data[coords]


   @property
   def dtype(self):
       return self._backend.dtype(self._data)

   @property 
   def ndim(self):
       return self._backend.ndim(self._data)

   @property
   def shape(self):
       return self._backend.shape(self._data)


   def __neg__(self):

       return ops.neg(self)


   def __add__(self, other):

       return ops.add(self, other)


   def __mul__(self, other):

       return ops.mul(self, other)


   def __radd__(self, other):

       if other == 0:
          return self

       return ops.add(self, asarray(self._backend, other))


   def __rmul__(self, other):

       if other == 1:
          return self

       return ops.mul(self, asarray(self._backend, other))




# --- Function call --------------------------------------------------------- #

class FunCall:

   def __init__(self, fun, content=util.Sequence()):

       self._fun     = fun
       self._content = content


   def attach(self, array, data):

       return self.__class__(self._content.push((array, data)))


   def size(self):

       return len(self._content)


   def execute(self):

       arrays, datas = zip(*self._content)
       space         = arrays[0].space() 

       return space.apply(self._fun, *datas) 




# --- Args ------------------------------------------------------------------ #

class Args:

   def __init__(self, *args):

       self._args = args


   def __len__(self):

       return len(self._args)


   def __contains__(self, x):

       return x in self._args


   def __iter__(self):

       return iter(self._args)


   def __getitem__(self, idx):

       return self._args[idx]


   def pluginto(self, funcall):

       for arg in self._args:
           funcall = arg.pluginto(funcall)

       return funcall.execute()




