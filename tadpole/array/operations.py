#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np

import tadpole.util     as util
import tadpole.autodiff as ad

import tadpole.array.core     as core
import tadpole.array.logical  as logical
import tadpole.array.function as function

from tadpole.array.function import (
   Args, 
   VisitCall, 
   SplitCall, 
   TransformCall,
)

from tadpole.array.core import (
   typecast_unary,
   typecast_binary,
)




###############################################################################
###                                                                         ###
###  Definitions of non-differentiable array operations                     ###
###                                                                         ###
###############################################################################


# --- Array value methods --------------------------------------------------- #

@ad.nondifferentiable
def allof(x, axis=None, **opts):

    def fun(backend, v): 
        return backend.all(v, axis, **opts)

    return Args(x).pluginto(TransformCall(fun))



@ad.nondifferentiable
def anyof(x, axis=None, **opts):

    def fun(backend, v): 
        return backend.any(v, axis, **opts)

    return Args(x).pluginto(TransformCall(fun))



@ad.nondifferentiable
def sign(x, **opts):

    def fun(backend, v): 
        return backend.sign(v, **opts)

    return Args(x).pluginto(TransformCall(fun))



@ad.nondifferentiable
def count_nonzero(x, axis=None, **opts):

    def fun(backend, v): 
        return backend.count_nonzero(v, axis, **opts)

    return Args(x).pluginto(TransformCall(fun))



@ad.nondifferentiable
def put(x, idxs, vals, accumulate=False): 

    def fun(backend, v):
        return backend.put(v, idxs, vals, accumulate=accumulate)

    return Args(x).pluginto(TransformCall(fun))



@ad.nondifferentiable
def argsort(x, axis=-1, **opts):

    def fun(backend, v):
        return backend.argsort(v, axis, **opts)

    return Args(x).pluginto(TransformCall(fun))


@ad.nondifferentiable
@typecast_unary
def iscomplex(x):

    def fun(backend, v):
        return backend.iscomplex(v)

    return Args(x).pluginto(VisitCall(fun))  




###############################################################################
###                                                                         ###
###  Definitions of differentiable array operations                         ###
###                                                                         ###
###############################################################################


# --- Array shape methods --------------------------------------------------- #

@ad.differentiable
def reshape(x, shape):

    def fun(backend, v):
        return backend.reshape(v, shape)

    return Args(x).pluginto(TransformCall(fun))



@ad.differentiable
def transpose(x, axes):

    def fun(backend, v):
        return backend.transpose(v, axes)

    return Args(x).pluginto(TransformCall(fun))



@ad.differentiable
def moveaxis(x, source, destination):

    def fun(backend, v): 
        return backend.moveaxis(v, source, destination)

    return Args(x).pluginto(TransformCall(fun))



@ad.differentiable
def squeeze(x, axis=None):

    def fun(backend, v): 
        return backend.squeeze(v, axis)

    return Args(x).pluginto(TransformCall(fun))



@ad.differentiable
def unsqueeze(x, axis):

    def fun(backend, v): 
        return backend.unsqueeze(v, axis)

    return Args(x).pluginto(TransformCall(fun))




# --- Array value methods --------------------------------------------------- #
 
@ad.differentiable
def amax(x, axis=None, **opts):

    def fun(backend, v): 
        return backend.max(v, axis, **opts)

    return Args(x).pluginto(TransformCall(fun))



@ad.differentiable
def amin(x, axis=None, **opts):

    def fun(backend, v): 
        return backend.min(v, axis, **opts)

    return Args(x).pluginto(TransformCall(fun))



@ad.differentiable
def absolute(x, **opts):

    def fun(backend, v): 
        return backend.abs(v, **opts)

    return Args(x).pluginto(TransformCall(fun))



@ad.differentiable
def flip(x, axis=None):

    def fun(backend, v): 
        return backend.flip(v, axis)

    return Args(x).pluginto(TransformCall(fun))



@ad.differentiable
def clip(x, minval, maxval, **opts):

    def fun(backend, v): 
        return backend.clip(v, minval, maxval, **opts)

    return Args(x).pluginto(TransformCall(fun))



@ad.differentiable
@typecast_binary
def where(condition, x, y):

    def fun(backend, cond_uv, u, v): 
        return backend.where(cond_uv, u, v)

    return Args(condition, x, y).pluginto(TransformCall(fun))




# --- Simple math operations ------------------------------------------------ #

@ad.differentiable
@typecast_unary
def conj(x, **opts):

    def fun(backend, v):
        return backend.conj(v, **opts)

    return Args(x).pluginto(TransformCall(fun))



@ad.differentiable
@typecast_unary
def real(x):

    def fun(backend, v):
        return backend.real(v)

    return Args(x).pluginto(TransformCall(fun))     



@ad.differentiable
@typecast_unary
def imag(x):

    def fun(backend, v):
        return backend.imag(v)

    return Args(x).pluginto(TransformCall(fun))
  


@ad.differentiable
@typecast_unary
def sqrt(x):

    def fun(backend, v):
        return backend.sqrt(v)

    return Args(x).pluginto(TransformCall(fun))



@ad.differentiable
@typecast_unary
def log(x):

    def fun(backend, v):
        return backend.log(v)

    return Args(x).pluginto(TransformCall(fun))



@ad.differentiable
@typecast_unary
def exp(x):

    def fun(backend, v):
        return backend.exp(v)

    return Args(x).pluginto(TransformCall(fun))



@ad.differentiable
@typecast_unary
def sin(x):

    def fun(backend, v):
        return backend.sin(v)

    return Args(x).pluginto(TransformCall(fun))



@ad.differentiable
@typecast_unary
def cos(x):

    def fun(backend, v):
        return backend.cos(v)

    return Args(x).pluginto(TransformCall(fun))



@ad.differentiable
@typecast_unary
def tan(x):

    def fun(backend, v):
        return backend.tan(v)

    return Args(x).pluginto(TransformCall(fun))



@ad.differentiable
@typecast_unary
def arcsin(x):

    def fun(backend, v):
        return backend.arcsin(v)

    return Args(x).pluginto(TransformCall(fun))



@ad.differentiable
@typecast_unary
def arccos(x):

    def fun(backend, v):
        return backend.arccos(v)

    return Args(x).pluginto(TransformCall(fun))



@ad.differentiable
@typecast_unary
def arctan(x):

    def fun(backend, v):
        return backend.arctan(v)

    return Args(x).pluginto(TransformCall(fun))



@ad.differentiable
@typecast_unary
def sinh(x):

    def fun(backend, v):
        return backend.sinh(v)

    return Args(x).pluginto(TransformCall(fun))



@ad.differentiable
@typecast_unary
def cosh(x):

    def fun(backend, v):
        return backend.cosh(v)

    return Args(x).pluginto(TransformCall(fun))



@ad.differentiable
@typecast_unary
def tanh(x):

    def fun(backend, v):
        return backend.tanh(v)

    return Args(x).pluginto(TransformCall(fun))



@ad.differentiable
@typecast_unary
def arcsinh(x):

    def fun(backend, v):
        return backend.arcsinh(v)

    return Args(x).pluginto(TransformCall(fun))



@ad.differentiable
@typecast_unary
def arccosh(x):

    def fun(backend, v):
        return backend.arccosh(v)

    return Args(x).pluginto(TransformCall(fun))



@ad.differentiable
@typecast_unary
def arctanh(x):

    def fun(backend, v):
        return backend.arctanh(v)

    return Args(x).pluginto(TransformCall(fun))



@ad.differentiable
@typecast_unary
def sumover(x, axis=None, dtype=None, **opts):

    def fun(backend, v):
        return backend.sumover(v, axis, dtype, **opts)

    return Args(x).pluginto(TransformCall(fun))



@ad.differentiable
@typecast_unary
def cumsum(x, axis=None, dtype=None, **opts):

    def fun(backend, v):
        return backend.cumsum(v, axis, dtype, **opts)

    return Args(x).pluginto(TransformCall(fun))




"""

# --- Linear algebra: multiplication methods -------------------------------- #

def einsum(equation, *xs, optimize=True):

    def fun(backend, *xs):
        return backend.einsum(equation, *xs, optimize=optimize)

    return Args(*xs).pluginto(TransformCall(fun))



def dot(x, y):

    def fun(backend, u, v):
        return backend.dot(u, v)

    return Args(x, y).pluginto(TransformCall(fun))



def kron(x, y):

    def fun(backend, u, v):
        return backend.kron(u, v)

    return Args(x, y).pluginto(TransformCall(fun))




# --- Linear algebra: decomposition methods --------------------------------- #

def svd(x):

    def fun(backend, v):
        return backend.svd(v)

    return Args(x).pluginto(SplitCall(fun))



def qr(x):

    def fun(backend, v):
        return backend.qr(v)

    return Args(x).pluginto(SplitCall(fun))



def eig(x):

    def fun(backend, v):
        return backend.eig(v)

    return Args(x).pluginto(SplitCall(fun))



def eigh(x):

    def fun(backend, v):
        return backend.eigh(v)

    return Args(x).pluginto(SplitCall(fun))
       



# --- Linear algebra: matrix exponential ------------------------------------ #

def expm(x):

    def fun(backend, v):
        return backend.expm(v)

    return Args(x).pluginto(TransformCall(fun))
       



# --- Linear algebra: misc methods ------------------------------------------ #

def htranspose(x, axes):

    def fun(backend, v):
        return backend.htranspose(v, axes)

    return Args(x).pluginto(TransformCall(fun))



def norm(x, order=None, axis=None, **opts):

    def fun(backend, v):
        return backend.htranspose(v, order, axis, **opts)

    return Args(x).pluginto(TransformCall(fun))

"""




