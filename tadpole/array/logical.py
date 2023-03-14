#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np

import tadpole.util     as util
import tadpole.autodiff as ad

import tadpole.array.function as function




###############################################################################
###                                                                         ###
###  Logical operations                                                     ###
###                                                                         ###
###############################################################################


# --- Approximate (close) equality ------------------------------------------ #

def close_opts(opts):

    rtol = opts.pop("rtol", 1e-5)
    atol = opts.pop("atol", 1e-8)

    return {"rtol": rtol, "atol": atol, **opts}



@ad.nondifferentiable
@typecast_binary
def allclose(x, y, **opts):

    def fun(backend, u, v):
        return backend.allclose(u, v, **close_opts(opts))

    return function.Args(x, y).pluginto(function.VisitCall(fun))



@ad.nondifferentiable
@typecast_binary
def isclose(x, y, **opts):
    
    def fun(backend, u, v):
        return backend.isclose(u, v, **close_opts(opts))

    return function.Args(x, y).pluginto(function.TransformCall(fun))




# --- Exact equality -------------------------------------------------------- #

@ad.nondifferentiable
@typecast_binary
def allequal(x, y):

    def fun(backend, u, v):
        return backend.allequal(u, v)

    return function.Args(x, y).pluginto(function.VisitCall(fun))



@ad.nondifferentiable
@typecast_binary
def isequal(x, y):

    def fun(backend, u, v):
        return backend.isequal(u, v)

    return function.Args(x, y).pluginto(function.TransformCall(fun))



@ad.nondifferentiable
@typecast_binary
def notequal(x, y):

    def fun(backend, u, v):
        return backend.notequal(u, v)

    return function.Args(x, y).pluginto(function.TransformCall(fun))




# --- Other logical operations ---------------------------------------------- #

@ad.nondifferentiable
@typecast_binary
def logical_and(x, y):

    def fun(backend, u, v):
        return backend.logical_and(u, v)

    return function.Args(x, y).pluginto(function.TransformCall(fun))



@ad.nondifferentiable
@typecast_binary
def logical_or(x, y):

    def fun(backend, u, v):
        return backend.logical_or(u, v)

    return function.Args(x, y).pluginto(function.TransformCall(fun))




