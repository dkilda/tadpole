#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np

from tadpole.array.array import Array, Args, FunCall




###############################################################################
###                                                                         ###
###  Definitions of specific array operations                               ###
###                                                                         ###
###############################################################################


# --- Array operations: unary ----------------------------------------------- #

def put(x, idxs, vals, accumulate=False):

    def fun(backend, v):
        return backend.put(v, idxs, vals, accumulate=accumulate)

    return Args(x).pluginto(FunCall(fun))


def reshape(x, shape):

    def fun(backend, v):
        return backend.reshape(v, shape)

    return Args(x).pluginto(FunCall(fun))


def neg(x):

    return Args(x).pluginto(FunCall(backend.neg))


def sin(x):

    return Args(x).pluginto(FunCall(backend.sin))


def cos(x):

    return Args(x).pluginto(FunCall(backend.cos))




# --- Array operations: binary ---------------------------------------------- #

def add(x, y):

    return Args(x, y).pluginto(FunCall(backend.add))


def mul(x, y):
        
    return Args(x, y).pluginto(FunCall(backend.mul))




# --- Array operations: nary ------------------------------------------------ #

def einsum(equation, *xs, optimize=True)

    def fun(backend, *xs):
        return backend.einsum(equation, *xs, optimize=optimize)

    return Args(*xs).pluginto(FunCall(fun))




