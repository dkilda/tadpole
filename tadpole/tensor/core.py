#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np

import tadpole.util     as util
import tadpole.autodiff as ad

import tadpole.tensor.backends as backends
import tadpole.tensor.function as function
import tadpole.tensor.grad     as grad

from tadpole.tensor.types import TensorLike, Pluggable




###############################################################################
###                                                                         ###
###  Helper functions                                                       ###
###                                                                         ###
###############################################################################


# --- Type cast for unary functions ----------------------------------------- #

def typecast_unary(fun):

    def wrap(x, *args, **kwargs):

        try:
            return fun(x, *args, **kwargs)       
 
        except (AttributeError, TypeError):

            return fun(astensor(x), *args, **kwargs)
         
    return wrap




# --- Type cast for binary functions ---------------------------------------- #

def typecast_binary(fun):

    def wrap(x, y, *args, **kwargs):

        try:
            return fun(x, y, *args, **kwargs)       
 
        except (AttributeError, TypeError):

            if not any(isinstance(v, Pluggable) for v in (x,y)):
               x = astensor(x)
               y = astensor(y) 

            if not isinstance(x, Pluggable):
               x = y.withdata(x) 

            if not isinstance(y, Pluggable):
               y = x.withdata(y) 

            return fun(x, y, *args, **kwargs)
         
    return wrap




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

       data = fun(inds.shape, *args, **opts)
       return Tensor(backend, data, inds)  




# --- Tensor factories (from shape) ------------------------------------------ #

@ad.differentiable
def sparse(inds, pos, vals, **opts):

    backend = backends.get_from(opts)

    if "dtype" in opts:
       vals = backend.asarray(vals)
       vals = backend.astype(vals, **opts)

    return grad.SparseGrad(
              backend, inds, pos, vals
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




# --- Tensor factories (from data) ------------------------------------------ #

def astensor(data, inds=None, **opts):

    if isinstance(data, TensorLike):
       return data

    backend = backends.get_from(opts)                            
    data    = backend.asarray(data, **opts)

    return Tensor(backend, data, inds)




# --- Tensor generators ----------------------------------------------------- #

@ad.nondifferentiable
def units(inds, dtype=None, backend=None):

    for pos in np.ndindex(*shape):
        yield unit(inds, pos, dtype=dtype, backend=backend)


@ad.nondifferentiable
def basis(inds, dtype=None, backend=None): 

    backend   = backends.get(backend)
    dtype     = backend.get_dtype(dtype)
    gen_units = units(inds, dtype=dtype, backend=backend)

    if  dtype in backend.complex_dtypes():

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

   def __init__(self, backend, inds, dtype):

       self._backend = backend
       self._dtype   = dtype
       self._inds    = inds


   # --- Comparisons --- #

   def __eq__(self, other):

       log = util.LogicalChain()

       log.typ(self, other)
       log.val(self._backend, other._backend)
       log.val(self._dtype,   other._dtype)
       log.val(self._inds,    other._inds)

       return bool(log)


   # --- Factories --- #

   def sparse(self, pos, vals):

       return self._create(sparse, pos, vals)


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
       return self._inds.size 

   @property 
   def ndim(self):
       return self._inds.ndim

   @property
   def shape(self):
       return self._inds.shape


   # --- Private helpers --- #

   def _create(self, fun, *args, **opts):

       return fun(
          self._inds, *args, 
          dtype=self._dtype, backend=self._backend, **opts
       )




###############################################################################
###                                                                         ###
###  Logical operations                                                     ###
###                                                                         ###
###############################################################################


# --- Approximate (close) equality ------------------------------------------ #

def close_opts(opts):

    rtol = opts.pop("rtol", 1e-5)
    atol = opts.pop("atol", 1e-8)

    return {"rtol": rtol, "atol": atol, **opts}



@ad.nondifferentiable
@typecast_binary
def allclose(x, y, **opts):

    def fun(backend, u, v):
        return backend.allclose(u, v, **close_opts(opts))

    return function.Args(x, y).pluginto(function.ExtractCall(fun))



@ad.nondifferentiable
@typecast_binary
def isclose(x, y, **opts):
    
    def fun(backend, u, v):
        return backend.isclose(u, v, **close_opts(opts))

    return function.Args(x, y).pluginto(function.ElemwiseCall(fun))




# --- Exact equality -------------------------------------------------------- #

@ad.nondifferentiable
@typecast_binary
def allequal(x, y):

    def fun(backend, u, v):
        return backend.allequal(u, v)

    return function.Args(x, y).pluginto(function.ExtractCall(fun))



@ad.nondifferentiable
@typecast_binary
def isequal(x, y):

    def fun(backend, u, v):
        return backend.isequal(u, v)

    return function.Args(x, y).pluginto(function.ElemwiseCall(fun))



@ad.nondifferentiable
@typecast_binary
def notequal(x, y):

    def fun(backend, u, v):
        return backend.notequal(u, v)

    return function.Args(x, y).pluginto(function.ElemwiseCall(fun))




# --- Other logical operations ---------------------------------------------- #

@ad.nondifferentiable
@typecast_binary
def logical_and(x, y):

    def fun(backend, u, v):
        return backend.logical_and(u, v)

    return function.Args(x, y).pluginto(function.ElemwiseCall(fun))



@ad.nondifferentiable
@typecast_binary
def logical_or(x, y):

    def fun(backend, u, v):
        return backend.logical_or(u, v)

    return function.Args(x, y).pluginto(function.ElemwiseCall(fun))




###############################################################################
###                                                                         ###
###  Tensor methods                                                         ###
###                                                                         ###
###############################################################################


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




# --- Tensor methods: gradient accumulation --------------------------------- #

@ad.differentiable
@typecast_binary
def addgrads(x, y):

    return y.addto(x)




# --- Tensor methods: arithmetics and element access ------------------------ # 

@ad.differentiable
def getitem(x, pos):

    def fun(backend, v):
        return v[pos]

    return function.Args(x).pluginto(function.TransformCall(fun))



@ad.differentiable
@typecast_unary
def neg(x):

    def fun(backend, v):
        return -v

    return function.Args(x).pluginto(function.ElemwiseCall(fun))



@ad.differentiable
@typecast_binary
def add(x, y):

    def fun(backend, u, v):
        return backend.add(u, v)

    return function.Args(x, y).pluginto(function.ElemwiseCall(fun))



@ad.differentiable
@typecast_binary
def sub(x, y):

    def fun(backend, u, v):
        return backend.sub(u, v)

    return function.Args(x, y).pluginto(function.ElemwiseCall(fun))



@ad.differentiable
@typecast_binary
def mul(x, y):

    def fun(backend, u, v):
        return backend.mul(u, v)

    return function.Args(x, y).pluginto(function.ElemwiseCall(fun))



@ad.differentiable
@typecast_binary
def div(x, y):

    def fun(backend, u, v):
        return backend.div(u, v)

    return function.Args(x, y).pluginto(function.ElemwiseCall(fun))



@ad.differentiable
@typecast_binary
def power(x, y):

    def fun(backend, u, v):
        return backend.power(u, v)

    return function.Args(x, y).pluginto(function.ElemwiseCall(fun))




###############################################################################
###                                                                         ###
###  Definition of tensor                                                   ###
###                                                                         ###
###############################################################################


# --- Tensor ---------------------------------------------------------------- #

class Tensor(TensorLike, Pluggable):

   # --- Construction --- #

   def __init__(self, backend, data, inds=None):

       if inds is None:
          inds = index.Indices()

       if not isinstance(backend, backends.backend.Backend):
          raise ValueError((
             f"Tensor: backend must be an instance "
             f"of Backend, but it is {backend}"
          ))  

       if data.shape != inds.shape,
          raise ValueError((
             f"Tensor: data and indices must have matching shapes, "
             f"but data shape {data.shape} != index shape {inds.shape}"
          ))

       self._backend = backend
       self._data    = data
       self._inds    = inds


   # --- Plugging into function calls --- #

   def pluginto(self, funcall):

       return funcall.attach(self._backend, self._data, self._inds)


   # --- Using in gradient accumulations --- #

   def addto(self, other):

       if not other:
          other = grad.ZeroGrad()

       if isinstance(other, grad.ZeroGrad): 
          return self

       if isinstance(other, grad.SparseGrad):
          return other.addto(self)

       assert self._inds == other._inds, (
          f"Tensor.addto(): "
          f"gradient accumulation cannot be performed for tensors "
          f"with non-matching indices {self._inds} != {other._inds}"
       )

       data = self._backend.add(self._data, other._data)
       return other.withdata(data)


   # --- Basic functionality --- #

   def copy(self, deep=True):

       data = self._backend.copy(self._data) if deep else self._data

       return self.__class__(self._backend, data, self._inds)


   def todense(self):

       return self


   def withdata(self, data):

       return astensor(data, self._inds, backend=self._backend)


   def space(self):

       return TensorSpace(self._backend, self._inds, self.dtype)


   def item(self, *pos): 

       return self._backend.item(self._data, *pos)


   # --- Tensor properties --- #

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

   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          log.val(self._backend, other._backend)
 
       if bool(log):
          return allequal(self, other)

       return False


   # --- Arithmetics and element access --- # 

   def __getitem__(self, pos):

       return getitem(self, pos)


   def __neg__(self):

       return neg(self)

 
   def __add__(self, other):

       return add(self, other)


   def __sub__(self, other):

       return sub(self, other)


   def __mul__(self, other):

       return mul(self, other)


   def __truediv__(self, other):

       return div(self, other)


   def __pow__(self, other):

       return power(self, other)


   def __radd__(self, other):

       return add(other, self)

 
   def __rsub__(self, other):

       return sub(other, self)


   def __rmul__(self, other):

       return mul(other, self)


   def __rtruediv__(self, other):

       return div(other, self)


   def __rpow__(self, other):

       return power(other, self)




