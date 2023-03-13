#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np

import tadpole.util             as util
import tadpole.autodiff         as ad
import tadpole.array.operations as op





def extend(x, target, axis):

    if target.ndim == 0:
       return x
      
    return unsqueeze(x, axis) + target.space().zeros()

        

def unreduce(x, target, axis=None):

    def fun(g):

        g1 = extend(g, target, axis)
        x1 = extend(x, target, axis)

        mask = equal(target, x1)
        
        return g1 * mask / op.sumover(mask, axis=axis, keepdims=True)

    return fun



def match_shape(x, target, default_axis=0):

    while ndim(x) > ndim(target):
       x = sumover(x, axis=default_axis)

    for axis, size in enumerate(target.shape):
        if size == 1:
           x = sumover(x, axis=axis, keepdims=True)

    return x



def match_type(x, target):

    if iscomplex(x) and not iscomplex(target):
       return real(x)

    if not iscomplex(x) and iscomplex(target):
       return x + 0j

    return x



def match(x, target, **opts):

    return match_type(match_shape(x, target, **opts), target)
    





###############################################################################
###                                                                         ###
###  Definitions of non-differentiable array operations                     ###
###                                                                         ###
###############################################################################


# --- Array member methods: basic functionality ----------------------------- #

copy     = ad.nondifferentiable(op.copy)
todense  = ad.nondifferentiable(op.todense)
withdata = ad.nondifferentiable(op.withdata)
space    = ad.nondifferentiable(op.space)
item     = ad.nondifferentiable(op.item)


# --- Array member methods: properties -------------------------------------- #

dtype = ad.nondifferentiable(op.dtype)
size  = ad.nondifferentiable(op.size)
ndim  = ad.nondifferentiable(op.ndim)
shape = ad.nondifferentiable(op.shape)


# --- Logical functions ----------------------------------------------------- # 

equal       = ad.nondifferentiable(op.equal)
not_equal   = ad.nondifferentiable(op.not_equal)
logical_and = ad.nondifferentiable(op.logical_and)

allequal = ad.nondifferentiable(op.allequal)
allclose = ad.nondifferentiable(op.allclose)

allallequal = ad.nondifferentiable(op.allallequal)
allallclose = ad.nondifferentiable(op.allallclose)


# --- Array value methods --------------------------------------------------- #

allof         = ad.nondifferentiable(op.allof)
anyof         = ad.nondifferentiable(op.anyof)
sign          = ad.nondifferentiable(op.sign)
count_nonzero = ad.nondifferentiable(op.count_nonzero)
put           = ad.nondifferentiable(op.put)
argsort       = ad.nondifferentiable(op.argsort)
iscomplex     = ad.nondifferentiable(op.iscomplex)



###############################################################################
###                                                                         ###
###  Definitions of differentiable array operations                         ###
###                                                                         ###
###############################################################################


# --- Array member methods: arithmetics and element access ------------------ # 

getitem = ad.differentiable(op.getitem)

neg = ad.differentiable(op.neg)
add = ad.differentiable(op.add)
sub = ad.differentiable(op.sub)
mul = ad.differentiable(op.mul)
div = ad.differentiable(op.div)

power  = ad.differentiable(op.power)


# --- Array methods: for gradient accumulation ------------------------------ #

addgrads = ad.differentiable(op.addgrads)


# --- Array shape methods --------------------------------------------------- #

reshape   = ad.differentiable(op.reshape)
transpose = ad.differentiable(op.transpose)
moveaxis  = ad.differentiable(op.moveaxis)
squeeze   = ad.differentiable(op.squeeze)
unsqueeze = ad.differentiable(op.unsqueeze)


# --- Array value methods --------------------------------------------------- #

amax = ad.differentiable(op.amax)
amin = ad.differentiable(op.amin)

absolute = ad.differentiable(op.absolute)

flip = ad.differentiable(op.flip)
clip = ad.differentiable(op.clip)

where = ad.differentiable(op.where)


# --- Simple math operations ------------------------------------------------ #

conj = ad.differentiable(op.conj)
real = ad.differentiable(op.real)
imag = ad.differentiable(op.imag)

sqrt = ad.differentiable(op.sqrt)
log  = ad.differentiable(op.log)
exp  = ad.differentiable(op.exp)

sin = ad.differentiable(op.sin)
cos = ad.differentiable(op.cos)
tan = ad.differentiable(op.tan)

arcsin = ad.differentiable(op.arcsin)
arccos = ad.differentiable(op.arccos)
arctan = ad.differentiable(op.arctan)

sinh = ad.differentiable(op.sinh)
cosh = ad.differentiable(op.cosh)
tanh = ad.differentiable(op.tanh)

arcsinh = ad.differentiable(op.arcsinh)
arccosh = ad.differentiable(op.arccosh)
arctanh = ad.differentiable(op.arctanh)

sumover = ad.differentiable(op.sumover)
cumsum  = ad.differentiable(op.cumsum)


# --- Linear algebra: multiplication methods -------------------------------- #

einsum = ad.differentiable(op.einsum)
dot    = ad.differentiable(op.dot)
kron   = ad.differentiable(op.kron)


# --- Linear algebra: decomposition methods --------------------------------- #

svd  = ad.differentiable(op.svd)
qr   = ad.differentiable(op.qr)
eig  = ad.differentiable(op.eig)
eigh = ad.differentiable(op.eigh)


# --- Linear algebra: matrix exponential ------------------------------------ #

expm = ad.differentiable(op.expm)


# --- Linear algebra: misc methods ------------------------------------------ #

norm = ad.differentiable(op.norm)


def htranspose(x, axes): 
    return transpose(conj(x), axes)















