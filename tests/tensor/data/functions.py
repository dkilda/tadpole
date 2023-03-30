#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import numpy as np

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.array    as ar
import tadpole.tensor   as tn
import tadpole.index    as tid

import tadpole.array.backends as backends

import tadpole.tensor.elemwise_unary  as tnu
import tadpole.tensor.elemwise_binary as tnb

import tests.tensor.fakes as fake
import tests.tensor.data  as data


from tadpole.tensor.types import (
   Pluggable,
   Tensor, 
   Space,
)


from tadpole.index import (
   Index,
   IndexGen,  
   Indices,
)




# --- Wrapped function data ------------------------------------------------- #

WrappedFunctionData = collections.namedtuple("WrappedFunctionData", [
                         "wrappedfun", "fun", "out", "args",
                      ])




def unary_wrappedfun_dat_001(backend):

    def fun(x):
        op = tnu.tensor_elemwise_unary(x)
        return op.sin()    

    backend = backends.get(backend)

    u   = backend.randn((1,), seed=1)[0]
    out = tn.TensorGen(ar.ArrayUnary(backend, backend.sin(u)))

    wrappedfun = tnu.typecast_unary(fun)
    return WrappedFunctionData(wrappedfun, fun, out, (u,))




def unary_wrappedfun_dat_002(backend):

    def fun(x):
        op = tnu.tensor_elemwise_unary(x)
        return op.sin()    

    x = data.tensor_dat(data.randn)(
           backend, ("i","j","k"), (2,3,4), dtype="complex128", seed=1
        )

    out = ar.ArrayUnary(x.backend, x.backend.sin(x.data))
    out = tn.TensorGen(out, x.inds)

    wrappedfun = tnu.typecast_unary(fun)
    return WrappedFunctionData(wrappedfun, fun, out, (x.tensor,))




def binary_wrappedfun_dat_001(backend):

    def fun(x, y):
        op = tnb.tensor_elemwise_binary(x, y)
        return op.mul()    

    backend = backends.get(backend)

    u = backend.randn((1,), seed=1)[0]
    v = backend.randn((1,), seed=2)[0]

    out = tn.TensorGen(ar.ArrayUnary(backend, backend.mul(u, v)))

    wrappedfun = tnb.typecast_binary(fun)
    return WrappedFunctionData(wrappedfun, fun, out, (u, v))




def binary_wrappedfun_dat_002(backend):

    def fun(x, y):
        op = tnb.tensor_elemwise_binary(x, y)
        return op.mul()    

    backend = backends.get(backend)

    x = data.tensor_dat(data.randn)(
           backend.name(), ("i","j","k"), (2,3,4), dtype="complex128", seed=1
        )
    v = x.backend.randn((1,), seed=2)[0]

    out = ar.ArrayUnary(x.backend, x.backend.mul(x.data, v))
    out = tn.TensorGen(out, x.inds)

    wrappedfun = tnb.typecast_binary(fun)
    return WrappedFunctionData(wrappedfun, fun, out, (x.tensor, v))




def binary_wrappedfun_dat_003(backend):

    def fun(x, y):
        op = tnb.tensor_elemwise_binary(x, y)
        return op.mul()    

    backend = backends.get(backend)

    x = data.tensor_dat(data.randn)(
           backend.name(), ("i","j","k"), (2,3,4), dtype="complex128", seed=1
        )
    v = x.backend.randn((1,), seed=2)[0]

    out = ar.ArrayUnary(x.backend, x.backend.mul(v, x.data))
    out = tn.TensorGen(out, x.inds)

    wrappedfun = tnb.typecast_binary(fun)
    return WrappedFunctionData(wrappedfun, fun, out, (v, x.tensor))




def binary_wrappedfun_dat_004(backend):

    def fun(x, y):
        op = tnb.tensor_elemwise_binary(x, y)
        return op.mul()    

    x = data.tensor_dat(data.randn)(
           backend, ("i","j","k"), (2,3,4), dtype="complex128", seed=1
        )
    y = data.tensor_dat(data.randn)(
           backend, ("i","j","k"), (2,3,4), dtype="complex128", seed=2
        )

    xtensor = tn.TensorGen(x.array, x.inds)
    ytensor = tn.TensorGen(y.array, x.inds)

    out = ar.ArrayUnary(x.backend, x.backend.mul(x.data, y.data))
    out = tn.TensorGen(out, x.inds)

    wrappedfun = tnb.typecast_binary(fun)
    return WrappedFunctionData(wrappedfun, fun, out, (xtensor, ytensor))




