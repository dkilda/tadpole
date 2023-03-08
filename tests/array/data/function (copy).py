#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import numpy as np

import tests.array.fakes as fake
import tests.array.data  as data

import tadpole.util       as util
import tadpole.array.core as core
import tadpole.array.grad as grad

import tadpole.array.backends   as backends
import tadpole.array.function   as function
import tadpole.array.operations as op




# --- Function data ------------------------------------------------------------ #

FunctionData = collections.namedtuple("FunctionData", [
                  "funcall",  "content", "args", 
                  "fun",      "out", 
                  "arrays",   "backends_and_datas",
               ])




def _function_dat(backend, shape, dtype, nargs, which="function"):

    xs = []
    for i in range(nargs):

        x = data.array_dat(data.randn)(
               backend, shape, dtype=dtype, seed=i+1
            )
        xs.append(x)
     
    arrays   = [x.array   for x in xs]
    backends = [x.backend for x in xs]
    datas    = [x.data    for x in xs]
    seq      = list(zip(backends, datas))

    if   which == "transform":
 
         out = data.array_dat(data.randn)(
                  backend, shape, dtype=dtype, seed=nargs+1
               ) 
         fun = fake.Fun(out.data, out.backend, *datas)
         out = util.Outputs(out.array)

         funcall = function.TransformCall(fun, util.Sequence(seq))


    elif which == "split":

         out1 = data.array_dat(data.randn)(
                   backend, shape, dtype=dtype, seed=nargs+1
                ) 
         out2 = data.array_dat(data.randn)(
                   backend, shape, dtype=dtype, seed=nargs+2
                ) 

         fun = fake.Fun((out1.data, out2.data), out1.backend, *datas)
         out = util.Outputs(out1.array, out2.array)

         funcall = function.SplitCall(fun, util.Sequence(seq))


    elif which == "visit":

         out = util.Outputs(fake.Value())
         fun = fake.Fun(out.unpack(), xs[0].backend, *datas)

         funcall = function.VisitCall(fun, util.Sequence(seq))


    else:
         raise ValueError(f"_function_dat: invalid input which {which}") 


    content = function.Content(util.Sequence(seq))
    args    = function.Args(*arrays)

    return FunctionData(
                        funcall,  content, args, 
                        fun,      out,
                        arrays,   seq,
                       )




def transform_dat(backend, nargs):

    return _function_dat(backend, (2,3,4), "complex128", nargs, "transform")




def split_dat(backend, nargs):

    return _function_dat(backend, (2,3,4), "complex128", nargs, "split")




def visit_dat(backend, nargs):

    return _function_dat(backend, (2,3,4), "complex128", nargs, "visit")




# --- Wrapped function data ------------------------------------------------------------ #

WrappedFunctionData = collections.namedtuple("WrappedFunctionData", [
                  "wrappedfun", "fun", "out", "args"
               ])




def unary_wrappedfun_dat_001(backend):

    def fun(x):

        def _fun(backend, v):
            return np.sin(v)

        out = function.Args(x).pluginto(
                 function.TransformCall(_fun)
              )
        return out.unpack()

    wrappedfun = op.typecast_unary(fun)

    np.random.seed(1)
    x   = np.random.randn(1)
    out = core.Array(backends.get(backend), np.sin(x))

    return WrappedFunctionData(wrappedfun, fun, out, (x,))




def unary_wrappedfun_dat_002(backend):

    def fun(x):

        def _fun(backend, v):
            return np.sin(v)

        out = function.Args(x).pluginto(
                 function.TransformCall(_fun)
              )
        return out.unpack()

    wrappedfun = op.typecast_unary(fun)

    x = data.array_dat(data.randn)(
           backend, (2,3,4), dtype="complex128", seed=1
        )

    out = core.Array(backends.get(backend), np.sin(x.data))

    return WrappedFunctionData(wrappedfun, fun, out, (x.array,))




def binary_wrappedfun_dat_001(backend):

    def fun(x, y):

        def _fun(backend, u, v):
            return u * v

        out = function.Args(x, y).pluginto(
                 function.TransformCall(_fun)
              )
        return out.unpack()

    wrappedfun = op.typecast_binary(fun)

    np.random.seed(1)
    x = np.random.randn(1)
    y = np.random.randn(2)

    out = core.Array(backends.get(backend), x * y)

    return WrappedFunctionData(wrappedfun, fun, out, (x, y))




def binary_wrappedfun_dat_002(backend):

    def fun(x, y):

        def _fun(backend, u, v):
            return u * v

        out = function.Args(x, y).pluginto(
                 function.TransformCall(_fun)
              )
        return out.unpack()

    wrappedfun = op.typecast_binary(fun)

    np.random.seed(1)
    x = np.random.randn(1)
    y = data.array_dat(data.randn)(
           backend, (2,3,4), dtype="complex128", seed=2
        )

    out = core.Array(backends.get(backend), x * y.data)

    return WrappedFunctionData(wrappedfun, fun, out, (x, y.array))




def binary_wrappedfun_dat_003(backend):

    def fun(x, y):

        def _fun(backend, u, v):
            return u * v

        out = function.Args(x, y).pluginto(
                 function.TransformCall(_fun)
              )
        return out.unpack()

    wrappedfun = op.typecast_binary(fun)

    x = data.array_dat(data.randn)(
           backend, (2,3,4), dtype="complex128", seed=1
        )

    np.random.seed(2)
    y = np.random.randn(1)

    out = core.Array(backends.get(backend), x.data * y)

    return WrappedFunctionData(wrappedfun, fun, out, (x.array, y))




def binary_wrappedfun_dat_004(backend):

    def fun(x, y):

        def _fun(backend, u, v):
            return u * v

        out = function.Args(x, y).pluginto(
                 function.TransformCall(_fun)
              )
        return out.unpack()

    wrappedfun = op.typecast_binary(fun)

    x = data.array_dat(data.randn)(
           backend, (2,3,4), dtype="complex128", seed=1
        )
    y = data.array_dat(data.randn)(
           backend, (2,3,4), dtype="complex128", seed=2
        )

    out = core.Array(backends.get(backend), x.data * y.data)

    return WrappedFunctionData(wrappedfun, fun, out, (x.array, y.array))




