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
def allof(x, inds=None, **opts):

    def fun(backend, data, axis): 
        return backend.all(data, axis, **opts)

    return Args(x).pluginto(ReduceCall(fun, inds))



@ad.nondifferentiable
def anyof(x, inds=None, **opts):

    def fun(backend, data, axis): 
        return backend.any(data, axis, **opts)

    return Args(x).pluginto(ReduceCall(fun, inds))



@ad.nondifferentiable
def count_nonzero(x, inds=None, **opts):

    def fun(backend, data, axis): 
        return backend.count_nonzero(v, axis, **opts)

    return Args(x).pluginto(ReduceCall(fun, inds))



@ad.nondifferentiable
def sign(x, **opts):

    def fun(backend, data): 
        return backend.sign(data, **opts)

    return Args(x).pluginto(ElemwiseCall(fun))



@ad.nondifferentiable
def put(x, pos, vals, accumulate=False): 

    def fun(backend, data):
        return backend.put(data, pos, vals, accumulate=accumulate)

    return Args(x).pluginto(ElemwiseCall(fun))



@ad.nondifferentiable
@typecast_unary
def iscomplex(x):

    def fun(backend, data):
        return backend.iscomplex(data)

    return Args(x).pluginto(ExtractCall(fun))  




###############################################################################
###                                                                         ###
###  Definitions of differentiable tensor operations                        ###
###                                                                         ###
###############################################################################


# --- Tensor shape methods -------------------------------------------------- #

@ad.differentiable
def reindex(x, indmap):

    def fun(inds):

        outinds = list(inds)

        for i, ind in enumerate(inds):
            try:
                outinds[i] = indmap[ind]
            except KeyError:
                pass

        return Indices(*outinds) 

    return Args(x).pluginto(ReindexCall(fun))




@ad.differentiable
def fuse(x, fusemap):

    if isinstance(fusemap, dict):
       fusemap = fusemap.items()

    def fun(inds):

        for inp, out in fusemap:

            if not isinstance(inp, Index):
               inp = Index(inp, sizeof(*inp))

            assert sizeof(*inp) == sizeof(out), (
               f"fuse(): sizes of "
               f"input {inp} and output {out} indices must match, "
               f"but {sizeof(*inp)} != {sizeof(out)}"
            )

            inds = inds.remove(*inp).add(out)

        return inds

    return Args(x).pluginto(ReshapeCall(fun))




@ad.differentiable
def split(x, splitmap):

    if isinstance(splitmap, dict):
       splitmap = splitmap.items()

    def fun(inds):

        for inp, out in splitmap:

            assert sizeof(inp) == sizeof(*out), (
               f"split(): sizes of "
               f"input {inp} and output {out} indices must match, "
               f"but {sizeof(inp)} != {sizeof(*out)}"
            )

            axis = inds.axis(inp)
            inds = inds.remove(inp).add(*out, axis=axis)

        return inds

    return Args(x).pluginto(ReshapeCall(fun))




@ad.differentiable
def transpose(x, *order):

    def fun(backend, data, inds):

        assert set(inds) == set(order),
           f"index.transpose(): input and output must contain the same "
           f"set of indices, but input {inds} does not match output {order}."

        outinds = Indices(*order)
        outdata = backend.transpose(data, inds.axes(*order))

        return core.Tensor(backend, data, order)

    return Args(x).pluginto(TransformCall(fun))




@ad.differentiable
def squeeze(x):

    def fun(backend, data, inds):

        singletons = (ind for ind in inds if len(ind) == 1)

        outinds = inds.remove(*singletons)
        outdata = backend.squeeze(data, inds.axes(*singletons))

        return core.Tensor(backend, outdata, outinds)

    return Args(x).pluginto(TransformCall(fun))




@ad.differentiable
def unsqueeze(x, names):

    def fun(backend, data, inds):

        singletons = (Index(name) for name in names)

        outinds = inds.add(*singletons)
        outdata = backend.unsqueeze(data, outinds.axes(*singletons))

        return core.Tensor(backend, outdata, outinds)

    return Args(x).pluginto(TransformCall(fun))




@ad.differentiable
def sumover(x, inds=None, dtype=None, **opts):

    def fun(backend, data, axis):
        return backend.sumover(data, axis, dtype, **opts)

    return Args(x).pluginto(ReduceCall(fun, inds))




@ad.differentiable
def cumsum(x, inds=None, dtype=None, **opts):

    def fun(backend, data, axis):
        return backend.cumsum(data, axis, dtype, **opts)

    return Args(x).pluginto(ReduceCall(fun, inds))




# --- Tensor value methods -------------------------------------------------- #

@ad.differentiable
def amax(x, inds=None, **opts):

    def fun(backend, data, axis): 
        return backend.max(data, axis, **opts)

    return Args(x).pluginto(ReduceCall(fun, inds))



@ad.differentiable
def amin(x, inds=None, **opts):

    def fun(backend, data, axis): 
        return backend.min(data, axis, **opts)

    return Args(x).pluginto(ReduceCall(fun, inds))



@ad.differentiable
def absolute(x, **opts):

    def fun(backend, data): 
        return backend.abs(data, **opts)

    return Args(x).pluginto(ElemwiseCall(fun))



@ad.differentiable
def flip(x, flipinds=None):

    def fun(backend, data, inds): 

        axes = None
        if flipinds is not None:
           axes = inds.axes(*flipinds)

        outdata = backend.flip(data, axes)
        return core.Tensor(backend, outdata, inds)

    return Args(x).pluginto(TransformCall(fun))



@ad.differentiable
def clip(x, minval, maxval, **opts):

    def fun(backend, data): 
        return backend.clip(data, minval, maxval, **opts)

    return Args(x).pluginto(ElemwiseCall(fun))



@ad.differentiable
@typecast_binary
def where(condition, x, y):

    def fun(backend, cond_uv, u, v): 
        return backend.where(cond_uv, u, v)

    return Args(condition, x, y).pluginto(ElemwiseCall(fun))




# --- Standard math operations ---------------------------------------------- #

@ad.differentiable
@typecast_unary
def conj(x, **opts):

    def fun(backend, v):
        return backend.conj(v, **opts)

    return Args(x).pluginto(ElemwiseCall(fun))



@ad.differentiable
@typecast_unary
def real(x):

    def fun(backend, v):
        return backend.real(v)

    return Args(x).pluginto(ElemwiseCall(fun))     



@ad.differentiable
@typecast_unary
def imag(x):

    def fun(backend, v):
        return backend.imag(v)

    return Args(x).pluginto(ElemwiseCall(fun))
  


@ad.differentiable
@typecast_unary
def sqrt(x):

    def fun(backend, v):
        return backend.sqrt(v)

    return Args(x).pluginto(ElemwiseCall(fun))



@ad.differentiable
@typecast_unary
def log(x):

    def fun(backend, v):
        return backend.log(v)

    return Args(x).pluginto(ElemwiseCall(fun))



@ad.differentiable
@typecast_unary
def exp(x):

    def fun(backend, v):
        return backend.exp(v)

    return Args(x).pluginto(ElemwiseCall(fun))



@ad.differentiable
@typecast_unary
def sin(x):

    def fun(backend, v):
        return backend.sin(v)

    return Args(x).pluginto(ElemwiseCall(fun))



@ad.differentiable
@typecast_unary
def cos(x):

    def fun(backend, v):
        return backend.cos(v)

    return Args(x).pluginto(ElemwiseCall(fun))



@ad.differentiable
@typecast_unary
def tan(x):

    def fun(backend, v):
        return backend.tan(v)

    return Args(x).pluginto(ElemwiseCall(fun))



@ad.differentiable
@typecast_unary
def arcsin(x):

    def fun(backend, v):
        return backend.arcsin(v)

    return Args(x).pluginto(ElemwiseCall(fun))



@ad.differentiable
@typecast_unary
def arccos(x):

    def fun(backend, v):
        return backend.arccos(v)

    return Args(x).pluginto(ElemwiseCall(fun))



@ad.differentiable
@typecast_unary
def arctan(x):

    def fun(backend, v):
        return backend.arctan(v)

    return Args(x).pluginto(ElemwiseCall(fun))



@ad.differentiable
@typecast_unary
def sinh(x):

    def fun(backend, v):
        return backend.sinh(v)

    return Args(x).pluginto(ElemwiseCall(fun))



@ad.differentiable
@typecast_unary
def cosh(x):

    def fun(backend, v):
        return backend.cosh(v)

    return Args(x).pluginto(ElemwiseCall(fun))



@ad.differentiable
@typecast_unary
def tanh(x):

    def fun(backend, v):
        return backend.tanh(v)

    return Args(x).pluginto(ElemwiseCall(fun))



@ad.differentiable
@typecast_unary
def arcsinh(x):

    def fun(backend, v):
        return backend.arcsinh(v)

    return Args(x).pluginto(ElemwiseCall(fun))



@ad.differentiable
@typecast_unary
def arccosh(x):

    def fun(backend, v):
        return backend.arccosh(v)

    return Args(x).pluginto(ElemwiseCall(fun))



@ad.differentiable
@typecast_unary
def arctanh(x):

    def fun(backend, v):
        return backend.arctanh(v)

    return Args(x).pluginto(ElemwiseCall(fun))




# --- Linear algebra: multiplication methods -------------------------------- #

@ad.differentiable
def einsum(*xs, outinds=None, optimize=True):

    def fun(backend, equation, *datas):
        return backend.einsum(equation, *datas, optimize=optimize)

    return Args(*xs).pluginto(EinsumCall(fun, outinds))



@ad.differentiable
def dot(x, y):

    def fun(backend, u, v):
        return backend.dot(u, v)

    return Args(x, y).pluginto(DotCall(fun))




# --- Linear algebra: decomposition methods --------------------------------- #

@ad.differentiable
def svd(x, mind, linds=None, rinds=None, **opts):

    def fun(backend, v):
        return backend.svd(v)

    return Args(x).pluginto(SplitCall(fun))



@ad.differentiable
def qr(x):

    def fun(backend, v):
        return backend.qr(v)

    return Args(x).pluginto(DoubleDecompCall(fun))



@ad.differentiable
def eig(x):

    def fun(backend, v):
        return backend.eig(v)

    return Args(x).pluginto(DoubleDecompCall(fun))



@ad.differentiable
def eigh(x):

    def fun(backend, v):
        return backend.eigh(v)

    return Args(x).pluginto(DoubleDecompCall(fun))
       












#############################################################
"""
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

"""












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




################################################################################
################################################################################
################################################################################
################################################################################
################################################################################





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







"""
"""



