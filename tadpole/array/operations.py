#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np

import tadpole.util as util

import tadpole.array.core     as core
import tadpole.array.logical  as logical
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


# --- Array member methods: basic functionality ----------------------------- #

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




# --- Array member methods: properties -------------------------------------- #

def dtype(x):
    return x.dtype


def size(x):
    return x.size


def ndim(x):
    return x.ndim


def shape(x):
    return x.shape




# --- Logical functions ----------------------------------------------------- # 

allequal = logical.allequal
allclose = logical.allclose

allallequal = logical.allallequal
allallclose = logical.allallclose



@typecast_binary
def equal(x, y):

    def fun(backend, u, v):
        return backend.equal(u, v)

    return Args(x, y).pluginto(TransformCall(fun))



@typecast_binary
def not_equal(x, y):

    def fun(backend, u, v):
        return backend.not_equal(u, v)

    return Args(x, y).pluginto(TransformCall(fun))



@typecast_binary
def logical_and(x, y):

    def fun(backend, u, v):
        return backend.logical_and(u, v)

    return Args(x, y).pluginto(TransformCall(fun))





# --- Array shape methods --------------------------------------------------- #




# --- Array value methods --------------------------------------------------- #

def allof(x, axis=None, **opts):

    def fun(backend, v): 
        return backend.all(v, axis, **opts)

    return Args(x).pluginto(TransformCall(fun))



def anyof(x, axis=None, **opts):

    def fun(backend, v): 
        return backend.any(v, axis, **opts)

    return Args(x).pluginto(TransformCall(fun))



def sign(x, **opts):

    def fun(backend, v): 
        return backend.sign(v, **opts)

    return Args(x).pluginto(TransformCall(fun))



def count_nonzero(x, axis=None, **opts):

    def fun(backend, v): 
        return backend.count_nonzero(v, axis, **opts)

    return Args(x).pluginto(TransformCall(fun))



def put(x, idxs, vals, accumulate=False): 

    def fun(backend, v):
        return backend.put(v, idxs, vals, accumulate=accumulate)

    return Args(x).pluginto(TransformCall(fun))



def argsort(x, axis=-1, **opts):

    def fun(backend, v):
        return backend.argsort(v, axis, **opts)

    return Args(x).pluginto(TransformCall(fun))


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


# --- Array member methods: arithmetics and element access ------------------ # 

def getitem(x, idx):
    return x[idx]


@typecast_unary
def neg(x):
    return -x


@typecast_binary
def add(x, y):
    return x + y


@typecast_binary
def sub(x, y):
    return x - y


@typecast_binary
def mul(x, y):
    return x * y


@typecast_binary
def div(x, y):
    return x / y


@typecast_binary
def power(x, y):
    return x**y




# --- Array methods: for gradient accumulation ------------------------------ #

@typecast_binary
def addgrads(x, y):

    return y.addto(x)




# --- Array shape methods --------------------------------------------------- #

def reshape(x, shape):

    def fun(backend, v):
        return backend.reshape(v, shape)

    return Args(x).pluginto(TransformCall(fun))



def transpose(x, axes):

    def fun(backend, v):
        return backend.transpose(v, axes)

    return Args(x).pluginto(TransformCall(fun))



def moveaxis(x, source, destination):

    def fun(backend, v): 
        return backend.moveaxis(v, source, destination)

    return Args(x).pluginto(TransformCall(fun))



def squeeze(x, axis=None):

    def fun(backend, v): 
        return backend.squeeze(v, axis)

    return Args(x).pluginto(TransformCall(fun))



def unsqueeze(x, axis):

    def fun(backend, v): 
        return backend.unsqueeze(v, axis)

    return Args(x).pluginto(TransformCall(fun))




# --- Array value methods --------------------------------------------------- #
 
def amax(x, axis=None, **opts):

    def fun(backend, v): 
        return backend.max(v, axis, **opts)

    return Args(x).pluginto(TransformCall(fun))



def amin(x, axis=None, **opts):

    def fun(backend, v): 
        return backend.min(v, axis, **opts)

    return Args(x).pluginto(TransformCall(fun))



def absolute(x, **opts):

    def fun(backend, v): 
        return backend.abs(v, **opts)

    return Args(x).pluginto(TransformCall(fun))



def flip(x, axis=None):

    def fun(backend, v): 
        return backend.flip(v, axis)

    return Args(x).pluginto(TransformCall(fun))



def clip(x, minval, maxval, **opts):

    def fun(backend, v): 
        return backend.clip(v, minval, maxval, **opts)

    return Args(x).pluginto(TransformCall(fun))



def where(condition, x, y):

    def fun(backend, cond_uv, u, v): 
        return backend.where(cond_uv, u, v)

    return Args(condition, x, y).pluginto(TransformCall(fun))




# --- Simple math operations ------------------------------------------------ #

@typecast_unary
def conj(x, **opts):

    def fun(backend, v):
        return backend.conj(v, **opts)

    return Args(x).pluginto(TransformCall(fun))



@typecast_unary
def real(x):

    def fun(backend, v):
        return backend.real(v)

    return Args(x).pluginto(TransformCall(fun))     



@typecast_unary
def imag(x):

    def fun(backend, v):
        return backend.imag(v)

    return Args(x).pluginto(TransformCall(fun))
  


@typecast_unary
def sqrt(x):

    def fun(backend, v):
        return backend.sqrt(v)

    return Args(x).pluginto(TransformCall(fun))



@typecast_unary
def log(x):

    def fun(backend, v):
        return backend.log(v)

    return Args(x).pluginto(TransformCall(fun))



@typecast_unary
def exp(x):

    def fun(backend, v):
        return backend.exp(v)

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



@typecast_unary
def tan(x):

    def fun(backend, v):
        return backend.tan(v)

    return Args(x).pluginto(TransformCall(fun))



@typecast_unary
def arcsin(x):

    def fun(backend, v):
        return backend.arcsin(v)

    return Args(x).pluginto(TransformCall(fun))



@typecast_unary
def arccos(x):

    def fun(backend, v):
        return backend.arccos(v)

    return Args(x).pluginto(TransformCall(fun))



@typecast_unary
def arctan(x):

    def fun(backend, v):
        return backend.arctan(v)

    return Args(x).pluginto(TransformCall(fun))



@typecast_unary
def sinh(x):

    def fun(backend, v):
        return backend.sinh(v)

    return Args(x).pluginto(TransformCall(fun))



@typecast_unary
def cosh(x):

    def fun(backend, v):
        return backend.cosh(v)

    return Args(x).pluginto(TransformCall(fun))



@typecast_unary
def tanh(x):

    def fun(backend, v):
        return backend.tanh(v)

    return Args(x).pluginto(TransformCall(fun))



@typecast_unary
def arcsinh(x):

    def fun(backend, v):
        return backend.arcsinh(v)

    return Args(x).pluginto(TransformCall(fun))



@typecast_unary
def arccosh(x):

    def fun(backend, v):
        return backend.arccosh(v)

    return Args(x).pluginto(TransformCall(fun))



@typecast_unary
def arctanh(x):

    def fun(backend, v):
        return backend.arctanh(v)

    return Args(x).pluginto(TransformCall(fun))



@typecast_unary
def sumover(x, axis=None, dtype=None, **opts):

    def fun(backend, v):
        return backend.sumover(v, axis, dtype, **opts)

    return Args(x).pluginto(TransformCall(fun))



@typecast_unary
def cumsum(x, axis=None, dtype=None, **opts):

    def fun(backend, v):
        return backend.cumsum(v, axis, dtype, **opts)

    return Args(x).pluginto(TransformCall(fun))




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













