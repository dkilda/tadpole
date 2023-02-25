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

def reshape(x, shape):

    def fun(backend, v):
        return backend.reshape(v, shape)

    return Args(x).pluginto(FunCall(fun))




# --- Array operations: binary ---------------------------------------------- #

def mul(x, y):

    def fun(backend, v, u):
        return backend.mul(v, u)
         
    return Args(x, y).pluginto(FunCall(fun))




# --- Array operations: nary ------------------------------------------------ #

def einsum(equation, *xs, optimize=True)

    def fun(backend, *xs):
        return backend.einsum(equation, *xs, optimize=optimize)

    return Args(*xs).pluginto(FunCall(fun))


