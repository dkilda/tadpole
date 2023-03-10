#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np

import tadpole.util as util

"""
Move the *implementations* of add/sub/mul/getitem to core.py and 
comparisons to comparisons.py: that way core.py and function.py 
won't depend on operations.py anymore! 

"""


import tadpole.array.grad     as grad # Fixes a circular import error
import tadpole.array.core     as core
import tadpole.array.function as function

from tadpole.array.function import (
   Args, 
   VisitCall, 
   SplitCall, 
   TransformCall,
)




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

            return fun(core.asarray(x), *args, **kwargs)
         
    return wrap




# --- Type cast for binary functions ---------------------------------------- #

def typecast_binary(fun):

    def wrap(x, y, *args, **kwargs):

        try:
            return fun(x, y, *args, **kwargs)       
 
        except (AttributeError, TypeError):

            if not any(isinstance(v, core.Pluggable) for v in (x,y)):
               x = core.asarray(x)
               y = core.asarray(y) 

            if not isinstance(x, core.Pluggable):
               x = y.withdata(x) 

            if not isinstance(y, core.Pluggable):
               y = x.withdata(y) 

            return fun(x, y, *args, **kwargs)
         
    return wrap




###############################################################################
###                                                                         ###
###  Definitions of non-differentiable array operations                     ###
###                                                                         ###
###############################################################################


# --- Basic array functionality --------------------------------------------- #

def copy(x, **opts):
    return x.copy(**opts)


def todense(x):
    return x.todense()


def withdata(x, data):
    return x.withdata(data)


def space(x):
    return x.space()


def item(x, *idx):
    return x.item(*idx)




# --- Array properties ------------------------------------------------------ #

def dtype(x):
    return x.dtype


def size(x):
    return x.size


def ndim(x):
    return x.ndim


def shape(x):
    return x.shape




# --- Array comparisons ----------------------------------------------------- # 

def allequal(x, y):

    return x.allequal(y)


def allclose(x, y, **opts):

    return x.allclose(y, **opts)


def allallequal(xs, ys):

    return all(allequal(x, y) for x, y in zip(xs, ys))


def allallclose(xs, ys, **opts):

    return all(allclose(x, y, **opts) for x, y in zip(xs, ys))




# --- Misc array operations ------------------------------------------------- #

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

def getitem(x, idx):

    def fun(backend, v):
        return v[idx]

    return Args(x).pluginto(TransformCall(fun))



def reshape(x, shape):

    def fun(backend, v):
        return backend.reshape(v, shape)

    return Args(x).pluginto(TransformCall(fun))



@typecast_unary
def neg(x):

    def fun(backend, v):
        return backend.neg(v)

    return Args(x).pluginto(TransformCall(fun))



@typecast_unary
def sin(x):

    def fun(backend, v):
        return backend.sin(v)

    return Args(x).pluginto(TransformCall(fun))



@typecast_unary
def cos(x):

    def fun(backend, v):
        return backend.cos(v)

    return Args(x).pluginto(TransformCall(fun))




# --- Array operations: binary (for gradient accumulation) ------------------ #

@typecast_binary
def addgrads(x, y):

    return y.addto(x)




# --- Array operations: binary ---------------------------------------------- #

@typecast_binary
def add(x, y):

    def fun(backend, v, u):
        return backend.add(v, u)

    return Args(x, y).pluginto(TransformCall(fun))



@typecast_binary
def sub(x, y):

    def fun(backend, v, u):
        return backend.sub(v, u)

    return Args(x, y).pluginto(TransformCall(fun))



@typecast_binary
def mul(x, y):

    def fun(backend, v, u):
        return backend.mul(v, u)
        
    return Args(x, y).pluginto(TransformCall(fun))




# --- Array operations: nary ------------------------------------------------ #

def einsum(equation, *xs, optimize=True):

    def fun(backend, *xs):
        return backend.einsum(equation, *xs, optimize=optimize)

    return Args(*xs).pluginto(TransformCall(fun))





