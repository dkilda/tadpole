#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np

import tadpole.util     as util
import tadpole.autodiff as ad

import tadpole.array.grad     as grad
import tadpole.array.core     as core
import tadpole.array.function as function

from tadpole.array.function import (
   Args, 
   VisitCall, 
   SplitCall, 
   TransformCall,
)

from tadpole.array.arraylike import (
   ArrayLike,
)




###############################################################################
###                                                                         ###
###  Helper functions                                                       ###
###                                                                         ###
###############################################################################


# --- Type cast for unary functions ----------------------------------------- #

def typecast_binary(fun):

    def wrap(x, *args, **kwargs):

        try:
            return fun(x, *args, **kwargs)       
 
        except (AttributeError, TypeError):

            return fun(core.asarray(x), *args, **kwargs)
         
    return wrap




# --- Type cast for binary functions ---------------------------------------- #

def typecast_binary(fun):

    def wrap(x, y, *args, **kwargs):

        try:
            return fun(x, y, *args, **kwargs)       
 
        except (AttributeError, TypeError):

            if not any(isinstance(v, ArrayLike) for v in (x,y)):
               x = core.asarray(x)
               y = core.asarray(y) 

            if not isinstance(x, ArrayLike):
               x = y.asarray(x) 

            if not isinstance(y, ArrayLike):
               y = x.asarray(y) 

            return fun(x, y, *args, **kwargs)
         
    return wrap




###############################################################################
###                                                                         ###
###  Definitions of non-differentiable array operations                     ###
###                                                                         ###
###############################################################################


"""
# --- Generic array operations ---------------------------------------------- #

@ad.nondifferentiable
def floor(x, n):
    return x // n

@ad.nondifferentiable
def equals(x, y):
    return x == y 

"""

# --- Array properties and basic functionality ------------------------------ #

@ad.nondifferentiable
def dtype(x):

    return util.Outputs(x.dtype)


@ad.nondifferentiable
def size(x):

    return util.Outputs(x.size)


@ad.nondifferentiable
def ndim(x):

    return util.Outputs(x.ndim)


@ad.nondifferentiable
def shape(x):

    return util.Outputs(x.shape)


@ad.nondifferentiable
def asarray(x, data):

    return util.Outputs(x.asarray(data))


@ad.nondifferentiable
def copy(x, **opts):

    return util.Outputs(x.copy(**opts))


@ad.nondifferentiable
def item(self, *idx):

    return util.Outputs(x.item(*idx))




# --- Generic array operations ---------------------------------------------- #

@ad.nondifferentiable
def put(x, idxs, vals, accumulate=False): 

    def fun(backend, v):
        return backend.put(v, idxs, vals, accumulate=accumulate)

    return Args(x).pluginto(TransformCall(fun))




###############################################################################
###                                                                         ###
###  Definitions of differentiable array operations                         ###
###                                                                         ###
###############################################################################


# --- Array operations: unary ----------------------------------------------- #

@ad.differentiable
def getitem(x, idx):

    def fun(backend, v):
        return v[idx]

    return Args(x).pluginto(TransformCall(fun))


@ad.differentiable
def reshape(x, shape):

    def fun(backend, v):
        return backend.reshape(v, shape)

    return Args(x).pluginto(TransformCall(fun))


@ad.differentiable
def neg(x):

    def fun(backend, v):
        return backend.neg(v)

    return Args(x).pluginto(TransformCall(fun))


@ad.differentiable
def sin(x):

    def fun(backend, v):
        return backend.sin(v)

    return Args(x).pluginto(TransformCall(fun))


@ad.differentiable
def cos(x):

    def fun(backend, v):
        return backend.cos(v)

    return Args(x).pluginto(TransformCall(fun))




# --- Array operations: binary ---------------------------------------------- #

@ad.differentiable
def add(x, y):

    def fun(backend, v, u):
        return backend.add(v, u)

    return Args(x, y).pluginto(TransformCall(fun))


@ad.differentiable
def sub(x, y):

    def fun(backend, v, u):
        return backend.sub(v, u)

    return Args(x, y).pluginto(TransformCall(fun))


@ad.differentiable
def mul(x, y):

    def fun(backend, v, u):
        return backend.mul(v, u)
        
    return Args(x, y).pluginto(TransformCall(fun))




# --- Gradient operations --------------------------------------------------- #

def gradadd(x, y):

    if isinstance(y, grad.SparseGrad): return sparseadd(x, y)
    if isinstance(x, grad.SparseGrad): return sparseadd(y, x)

    if isinstance(x, grad.ZeroGrad): return y
    if isinstance(y, grad.ZeroGrad): return x

    return add(x, y)
       

@ad.differentiable
def sparseadd(x, y):

    return x.sparseadd(y)




# --- Array operations: nary ------------------------------------------------ #

@ad.differentiable
def einsum(equation, *xs, optimize=True):

    def fun(backend, *xs):
        return backend.einsum(equation, *xs, optimize=optimize)

    return Args(*xs).pluginto(TransformCall(fun))




