#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np

import tadpole.util     as util
import tadpole.autodiff as ad

import tadpole.array.core     as core
import tadpole.array.function as function

from tadpole.array.function import (
   Args, 
   VisitCall, 
   SplitCall, 
   TransformCall,
)




"""
###############################################################################
###                                                                         ###
###  Definitions of non-differentiable array operations                     ###
###                                                                         ###
###############################################################################


# --- Generic array operations ---------------------------------------------- #

@ad.nondifferentiable
def floor(x, n):
    return x // n

@ad.nondifferentiable
def equals(x, y):
    return x == y 

"""


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
def put(x, idxs, vals, accumulate=False):

    def fun(backend, v):
        return backend.put(v, idxs, vals, accumulate=accumulate)

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

"""
@ad.differentiable
def grad_add(x, y):

    def fun(backend):

        


       if other == 0:
          other = self._space.zeros()

       return op.put(other, self._idxs, self._vals, accumulate=True)




    def fun(backend, v, u):
        return backend.add(v, u)

    return Args(x, y).pluginto(TransformCall(fun))
"""


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




# --- Array operations: nary ------------------------------------------------ #

@ad.differentiable
def einsum(equation, *xs, optimize=True):

    def fun(backend, *xs):
        return backend.einsum(equation, *xs, optimize=optimize)

    return Args(*xs).pluginto(TransformCall(fun))




