#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np
import tadpole.autodiff.graph as tdgraph

from tadpole.array.array import Array, Args, FunCall




###############################################################################
###                                                                         ###
###  Definitions of specific array operations                               ###
###                                                                         ###
###############################################################################


# --- Array operations: unary ----------------------------------------------- #

@tdgraph.differentiable
def get(x, idx):

    def fun(backend, v):
        return v[idx]

    return Args(x).pluginto(FunCall(fun))


@tdgraph.differentiable
def put(x, idxs, vals, accumulate=False):

    def fun(backend, v):
        return backend.put(v, idxs, vals, accumulate=accumulate)

    return Args(x).pluginto(FunCall(fun))


@tdgraph.differentiable
def reshape(x, shape):

    def fun(backend, v):
        return backend.reshape(v, shape)

    return Args(x).pluginto(FunCall(fun))


@tdgraph.differentiable
def neg(x):

    def fun(backend, v):
        return backend.neg(v)

    return Args(x).pluginto(FunCall(fun))


@tdgraph.differentiable
def sin(x):

    def fun(backend, v):
        return backend.sin(v)

    return Args(x).pluginto(FunCall(fun))


@tdgraph.differentiable
def cos(x):

    def fun(backend, v):
        return backend.cos(v)

    return Args(x).pluginto(FunCall(fun))




# --- Array operations: binary ---------------------------------------------- #

@tdgraph.differentiable
def add(x, y):

    def fun(backend, v, u):
        return backend.add(v, u)

    return Args(x, y).pluginto(FunCall(fun))


@tdgraph.differentiable
def mul(x, y):

    def fun(backend, v, u):
        return backend.mul(v, u)
        
    return Args(x, y).pluginto(FunCall(fun))




# --- Array operations: nary ------------------------------------------------ #

@tdgraph.differentiable
def einsum(equation, *xs, optimize=True)

    def fun(backend, *xs):
        return backend.einsum(equation, *xs, optimize=optimize)

    return Args(*xs).pluginto(FunCall(fun))




