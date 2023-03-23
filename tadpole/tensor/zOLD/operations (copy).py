#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np

import tadpole.util     as util
import tadpole.autodiff as ad

import tadpole.tensor.core     as core
import tadpole.tensor.function as function

from tadpole.tensor.function import (
   Args, 
   VisitCall, 
   SplitCall, 
   TransformCall,
)

from tadpole.tensor.core import (
   typecast_unary,
   typecast_binary,
)




###############################################################################
###                                                                         ###
###  Definitions of non-differentiable tensor operations                    ###
###                                                                         ###
###############################################################################


# --- Tensor value methods -------------------------------------------------- #

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
###  Definitions of differentiable tensor operations                        ###
###                                                                         ###
###############################################################################



# --- Unary, couples data and inds (acts on data but inds used as options) --- #

def amax(x, ind=None, **opts):

    def fun(backend, inds, v):
 
        axis = None if ind is None else inds.axis(ind)
        data = backend.max(v, axis, **opts)

        return core.Tensor(backend, data, tuple())

    return Args(x).pluginto(TransformCall(fun))



def transpose(x, *order):

    def fun(backend, inds, v):

        inds = index.transpose(inds, *order)
        data = backend.transpose(v, tuple(map(inds.axis, order)))

        return core.Tensor(backend, data, inds)

    return Args(x).pluginto(TransformCall(fun))



def sumover(x, *suminds):

    def fun(backend, inds, v):

        axes = tuple(map(inds.axis, suminds))
        inds = inds.remove(suminds)
        data = backend.sumover(v, axes)

        return core.Tensor(backend, data, inds)

    return Args(x).pluginto(TransformCall(fun))   



def reindex(x, indmap):

    def fun(backend, inds, v):

        inds = index.reindex(inds, indmap)

        assert v.shape == index.shapeof(*inds)

        return core.Tensor(backend, v, inds)

    return Args(x).pluginto(TransformCall(fun))




# --- Unary, data-only --- #

def sin(x):

    def fun(backend, inds, v):
        data = backend.sin(v)
        return core.Tensor(backend, data, inds)

    return Args(x).pluginto(TransformCall(fun))




def conj(x, **opts):

    def fun(backend, inds, v):

        data = backend.conj(v, **opts)
        return core.Tensor(backend, data, inds)

    return Args(x).pluginto(TransformCall(fun))




# --- Unary, inds-only --- #

def unsqueeze(x, names):

    def fun(backend, inds, v):

        inds  = index.unsqueeze(inds, names)
        shape = index.shapeof(*inds)
        data  = backend.reshape(v, shape) 

        return core.Tensor(backend, data, inds)

    return Args(x).pluginto(TransformCall(fun))   




def squeeze(x):

    def fun(backend, inds, v):

        inds  = index.squeeze(inds)
        shape = index.shapeof(*inds)
        data  = backend.reshape(v, shape)

        return core.Tensor(backend, data, inds)

    return Args(x).pluginto(TransformCall(fun))




def split(x, splitmap):

    def fun(backend, inds, v):

        inds  = index.split(inds, splitmap)
        shape = index.shapeof(*inds) 
        data  = backend.reshape(v, shape)

        return core.Tensor(backend, data, inds)

    return Args(x).pluginto(TransformCall(fun))




def fuse(x, fusemap):

    def fun(backend, inds, v):

        inds  = index.fuse(inds, fusemap)
        shape = index.shapeof(*inds) 
        data  = backend.reshape(v, shape)

        return core.Tensor(backend, data, inds)

    return Args(x).pluginto(TransformCall(fun))




# --- Binary elementwise --- #

def add(x, y):

    def fun(backend, inds, u, v):

        assert inds[0] == inds[1]
        data = backend.add(u, v)

        return core.Tensor(backend, data, inds[0])

    return function.Args(x, y).pluginto(function.TransformCall(fun))



# --- Unary decompose --- #

def svd(x):

    def fun(backend, v):
        return backend.svd(v) # TODO def all decomp steps in DecompCall, we only inject method (svd/qr/etc)

    return Args(x).pluginto(DecompCall(fun)) # Perhaps make DoubleDecomp / TripleDecomp? (DecompDouble, DecompTriple) 








# --- Tensor shape methods -------------------------------------------------- #

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




# --- Tensor value methods -------------------------------------------------- #
 
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






# --- Linear algebra: multiplication methods -------------------------------- #

@ad.differentiable
def einsum(equation, *xs, optimize=True):

    def fun(backend, *xs):
        return backend.einsum(equation, *xs, optimize=optimize)

    return Args(*xs).pluginto(TransformCall(fun))



@ad.differentiable
def dot(x, y):

    def fun(backend, u, v):
        return backend.dot(u, v)

    return Args(x, y).pluginto(TransformCall(fun))



@ad.differentiable
def kron(x, y):

    def fun(backend, u, v):
        return backend.kron(u, v)

    return Args(x, y).pluginto(TransformCall(fun))




# --- Linear algebra: decomposition methods --------------------------------- #

@ad.differentiable
def svd(x):

    def fun(backend, v):
        return backend.svd(v)

    return Args(x).pluginto(SplitCall(fun))



@ad.differentiable
def qr(x):

    def fun(backend, v):
        return backend.qr(v)

    return Args(x).pluginto(SplitCall(fun))



@ad.differentiable
def eig(x):

    def fun(backend, v):
        return backend.eig(v)

    return Args(x).pluginto(SplitCall(fun))



@ad.differentiable
def eigh(x):

    def fun(backend, v):
        return backend.eigh(v)

    return Args(x).pluginto(SplitCall(fun))
       



# --- Linear algebra: matrix exponential ------------------------------------ #

@ad.differentiable
def expm(x):

    def fun(backend, v):
        return backend.expm(v)

    return Args(x).pluginto(TransformCall(fun))
       



# --- Linear algebra: misc methods ------------------------------------------ #

def htranspose(x, axes):

    return transpose(conj(x), axes)



@ad.differentiable
def norm(x, order=None, axis=None, **opts):

    def fun(backend, v):
        return backend.htranspose(v, order, axis, **opts)

    return Args(x).pluginto(TransformCall(fun))

"""
"""



