#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.array    as ar

import tadpole.tensor.funcall as fn

from tadpole.tensor.types import TensorLike, Pluggable




###############################################################################
###                                                                         ###
###  Typecasting                                                            ###
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

    def fun(u, v):
        return ar.allclose(u, v, **close_opts(opts))

    return fn.Args(x, y).pluginto(fn.Extract(fun))



@ad.nondifferentiable
@typecast_binary
def isclose(x, y, **opts):
    
    def fun(u, v):
        return ar.isclose(u, v, **close_opts(opts))

    return fn.Args(x, y).pluginto(fn.Elemwise(fun))




# --- Exact equality -------------------------------------------------------- #

@ad.nondifferentiable
@typecast_binary
def allequal(x, y):

    def fun(u, v):
        return ar.allequal(u, v)

    return fn.Args(x, y).pluginto(fn.Extract(fun))



@ad.nondifferentiable
@typecast_binary
def isequal(x, y):

    def fun(u, v):
        return ar.isequal(u, v)

    return fn.Args(x, y).pluginto(fn.Elemwise(fun))



@ad.nondifferentiable
@typecast_binary
def notequal(x, y):

    def fun(u, v):
        return ar.notequal(u, v)

    return fn.Args(x, y).pluginto(fn.Elemwise(fun))




# --- Other logical operations ---------------------------------------------- #

@ad.nondifferentiable
@typecast_binary
def logical_and(x, y):

    def fun(u, v):
        return ar.logical_and(u, v)

    return fn.Args(x, y).pluginto(fn.Elemwise(fun))



@ad.nondifferentiable
@typecast_binary
def logical_or(x, y):

    def fun(u, v):
        return ar.logical_or(u, v)

    return fn.Args(x, y).pluginto(fn.Elemwise(fun))




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




# --- Tensor methods: element access ---------------------------------------- #

@ad.differentiable
def getitem(x, pos):

    def fun(u):
        return u[pos] # TODO impl getitem

    return fn.Args(x).pluginto(fn.Transform(fun))




# --- Tensor methods: arithmetics ------------------------------------------- # 

@ad.differentiable
@typecast_unary
def neg(x):

    def fun(u):
        return ar.neg(u)

    return fn.Args(x).pluginto(fn.Elemwise(fun))




@ad.differentiable
@typecast_binary
def add(x, y):

    def fun(u, v):
        return ar.add(u, v)

    return fn.Args(x, y).pluginto(fn.Elemwise(fun))




@ad.differentiable
@typecast_binary
def sub(x, y):

    def fun(u, v):
        return ar.sub(u, v)

    return fn.Args(x, y).pluginto(fn.Elemwise(fun))




@ad.differentiable
@typecast_binary
def mul(x, y):

    def fun(u, v):
        return ar.mul(u, v)

    return fn.Args(x, y).pluginto(fn.Elemwise(fun))




@ad.differentiable
@typecast_binary
def div(x, y):

    def fun(u, v):
        return ar.div(u, v)

    return fn.Args(x, y).pluginto(fn.Elemwise(fun))




@ad.differentiable
@typecast_binary
def power(x, y):

    def fun(u, v):
        return ar.power(u, v)

    return fn.Args(x, y).pluginto(fn.Elemwise(fun))




# --- Tensor methods: gradient accumulation --------------------------------- #

@ad.differentiable
@typecast_binary
def addgrads(x, y):

    return y.addto(x)




###############################################################################
###                                                                         ###
###  Definition of tensor                                                   ###
###                                                                         ###
###############################################################################


# --- Tensor ---------------------------------------------------------------- #

class Tensor(TensorLike, Pluggable):

   # --- Construction --- #

   def __init__(self, data, inds=None):

       if inds is None:
          inds = Indices()

       if data.shape != inds.shape,
          raise ValueError((
             f"{type(self).__name__}: 
             f"data and indices must have matching shapes, "
             f"but data shape {data.shape} != index shape {inds.shape}"
          ))

       self._data = data
       self._inds = inds


   # --- Plugging into function calls --- #

   def pluginto(self, funcall):

       return funcall.attach(self._data, self._inds)


   # --- Using in gradient accumulations --- #

   def addto(self, other):

       if not other:
          other = ZeroGrad()

       if isinstance(other, ZeroGrad): 
          return self

       if isinstance(other, SparseGrad):
          return other.addto(self)

       assert self._inds == other._inds, (
          f"{type(self).__name__}.addto: "
          f"gradient accumulation cannot be performed for tensors "
          f"with non-matching indices {self._inds} != {other._inds}"
       )

       data = ar.add(self._data, other._data)

       return other.withdata(data)


   # --- Basic functionality --- #

   def copy(self, virtual=False, **opts):

       data = self._data if virtual else self._data.copy(**opts)

       return self.__class__(data, self._inds)


   def todense(self):

       return self


   def withdata(self, data):

       return astensor(data, self._inds)


   def space(self):

       return TensorSpace(self._data.space(), self._inds) # TODO Impl ArraySpace


   def item(self, *pos): 

       return self._data.item(*pos)


   # --- Tensor properties --- #

   @property
   def dtype(self):
       return ar.dtype(self._data)

   @property 
   def size(self):
       return ar.size(self._data)  

   @property 
   def ndim(self):
       return ar.ndim(self._data)  

   @property
   def shape(self):
       return ar.shape(self._data)


   # --- Comparisons --- #

   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          log.val(self._inds, other._inds)
 
       if bool(log):
          return allequal(self._data, other._data)

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




