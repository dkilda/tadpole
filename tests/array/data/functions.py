#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import numpy as np

import tests.array.fakes as fake
import tests.array.data  as data

import tadpole.util           as util
import tadpole.array.unary    as unary
import tadpole.array.binary   as binary
import tadpole.array.backends as backends





# --- Wrapped function data ------------------------------------------------- #

WrappedFunctionData = collections.namedtuple("WrappedFunctionData", [
                         "wrappedfun", "fun", "out", "args"
                      ])




def unary_wrappedfun_dat_001(backend):

    backend = backends.get(backend)
    seed    = 1
    np.random.seed(seed)

    u   = np.random.randn(seed)
    x   = unary.Array(backend, u)            
    out = unary.Array(backend, np.sin(u))  

    fun        = fake.Fun(out, x)
    wrappedfun = unary.typecast(fun)

    return WrappedFunctionData(wrappedfun, fun, out, (u,))




def unary_wrappedfun_dat_002(backend):

    backend = backends.get(backend)

    x = data.array_dat(data.randn)(
           backend.name(), (2,3,4), dtype="complex128", seed=1
        )
    out = unary.Array(backend, np.sin(x.data))

    fun        = fake.Fun(out, x.array)
    wrappedfun = unary.typecast(fun)

    return WrappedFunctionData(wrappedfun, fun, out, (x.array,))




def binary_wrappedfun_dat_001(backend):

    backend = backends.get(backend)
    seed    = 1
    np.random.seed(seed)

    u = np.random.randn(seed)
    v = np.random.randn(seed+1)

    x   = unary.Array(backend, u)
    y   = unary.Array(backend, v)
    out = unary.Array(backend, u * v)

    fun        = fake.Fun(out, x, y)
    wrappedfun = binary.typecast(fun)

    return WrappedFunctionData(wrappedfun, fun, out, (u, v))



def binary_wrappedfun_dat_002(backend):

    backend = backends.get(backend)
    seed    = 1
    np.random.seed(seed)

    xdata = np.random.randn(seed)
    x     = unary.Array(backend, xdata)

    y = data.array_dat(data.randn)(
           backend.name(), (2,3,4), dtype="complex128", seed=seed+1
        )
    out = unary.Array(backend, xdata * y.data)

    fun        = fake.Fun(out, x, y.array)
    wrappedfun = binary.typecast(fun)

    return WrappedFunctionData(wrappedfun, fun, out, (xdata, y.array))




def binary_wrappedfun_dat_003(backend):

    backend = backends.get(backend)
    seed    = 1
    np.random.seed(seed)

    ydata = np.random.randn(seed)
    y     = unary.Array(backend, ydata)

    x = data.array_dat(data.randn)(
           backend.name(), (2,3,4), dtype="complex128", seed=seed+1
        )
    out = unary.Array(backend, x.data * ydata)

    fun        = fake.Fun(out, x.array, y)
    wrappedfun = binary.typecast(fun)

    return WrappedFunctionData(wrappedfun, fun, out, (x.array, ydata))




def binary_wrappedfun_dat_004(backend):

    backend = backends.get(backend)

    x = data.array_dat(data.randn)(
           backend.name(), (2,3,4), dtype="complex128", seed=1
        )
    y = data.array_dat(data.randn)(
           backend.name(), (2,3,4), dtype="complex128", seed=2
        )
    out = unary.Array(backend, x.data * y.data)

    fun        = fake.Fun(out, x.array, y.array)
    wrappedfun = binary.typecast(fun)

    return WrappedFunctionData(wrappedfun, fun, out, (x.array, y.array))




