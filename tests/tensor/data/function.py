#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import numpy as np

import tests.tensor.fakes as fake
import tests.tensor.data  as data

import tadpole.util        as util
import tadpole.tensor.core as core
import tadpole.tensor.grad as grad

import tadpole.tensor.backends   as backends
import tadpole.tensor.function   as function
import tadpole.tensor.operations as op




# --- Function data --------------------------------------------------------- #

FunctionData = collections.namedtuple("FunctionData", [
                  "funcall", "content", "args", 
                  "fun",     "out", 
                  "tensors", "backends_and_datas",
               ])




def _function_dat(backend, shape, dtype, nargs, which="function"):

    xs = []
    for i in range(nargs):

        x = data.tensor_dat(data.randn)(
               backend, shape, dtype=dtype, seed=i+1
            )
        xs.append(x)
     
    tensors  = [x.tensor   for x in xs]
    backends = [x.backend for x in xs]
    datas    = [x.data    for x in xs]
    seq      = list(zip(backends, datas))

    if   which == "transform":
 
         out = data.tensor_dat(data.randn)(
                  backend, shape, dtype=dtype, seed=nargs+1
               ) 
         fun = fake.Fun(out.data, out.backend, *datas)
         out = out.tensor

         funcall = function.TransformCall(fun, util.Sequence(seq))


    elif which == "split":

         out1 = data.tensor_dat(data.randn)(
                   backend, shape, dtype=dtype, seed=nargs+1
                ) 
         out2 = data.tensor_dat(data.randn)(
                   backend, shape, dtype=dtype, seed=nargs+2
                ) 

         fun = fake.Fun((out1.data, out2.data), out1.backend, *datas)
         out = (out1.tensor, out2.tensor)

         funcall = function.SplitCall(fun, util.Sequence(seq))


    elif which == "visit":

         out = fake.Value()
         fun = fake.Fun(out, xs[0].backend, *datas)

         funcall = function.VisitCall(fun, util.Sequence(seq))


    else:
         raise ValueError(f"_function_dat: invalid input which {which}") 


    content = function.Content(util.Sequence(seq))
    args    = function.Args(*tensors)

    return FunctionData(
                        funcall,  content, args, 
                        fun,      out,
                        tensors,  seq,
                       )




def transform_dat(backend, nargs):

    return _function_dat(backend, (2,3,4), "complex128", nargs, "transform")




def split_dat(backend, nargs):

    return _function_dat(backend, (2,3,4), "complex128", nargs, "split")




def visit_dat(backend, nargs):

    return _function_dat(backend, (2,3,4), "complex128", nargs, "visit")




# --- Wrapped function data ------------------------------------------------- #

WrappedFunctionData = collections.namedtuple("WrappedFunctionData", [
                         "wrappedfun", "fun", "out", "args"
                      ])




def unary_wrappedfun_dat_001(backend):

    def fun(x):

        def _fun(backend, v):
            return np.sin(v)

        return function.Args(x).pluginto(
                  function.TransformCall(_fun)
               )

    wrappedfun = op.typecast_unary(fun)

    np.random.seed(1)
    x   = np.random.randn(1)
    out = core.Tensor(backends.get(backend), np.sin(x))

    return WrappedFunctionData(wrappedfun, fun, out, (x,))




def unary_wrappedfun_dat_002(backend):

    def fun(x):

        def _fun(backend, v):
            return np.sin(v)

        return function.Args(x).pluginto(
                  function.TransformCall(_fun)
               )

    wrappedfun = op.typecast_unary(fun)

    x = data.tensor_dat(data.randn)(
           backend, (2,3,4), dtype="complex128", seed=1
        )

    out = core.Tensor(backends.get(backend), np.sin(x.data))

    return WrappedFunctionData(wrappedfun, fun, out, (x.tensor,))




def binary_wrappedfun_dat_001(backend):

    def fun(x, y):

        def _fun(backend, u, v):
            return u * v

        return function.Args(x, y).pluginto(
                  function.TransformCall(_fun)
               )

    wrappedfun = op.typecast_binary(fun)

    np.random.seed(1)
    x = np.random.randn(1)
    y = np.random.randn(2)

    out = core.Tensor(backends.get(backend), x * y)

    return WrappedFunctionData(wrappedfun, fun, out, (x, y))




def binary_wrappedfun_dat_002(backend):

    def fun(x, y):

        def _fun(backend, u, v):
            return u * v

        return function.Args(x, y).pluginto(
                  function.TransformCall(_fun)
               )

    wrappedfun = op.typecast_binary(fun)

    np.random.seed(1)
    x = np.random.randn(1)
    y = data.tensor_dat(data.randn)(
           backend, (2,3,4), dtype="complex128", seed=2
        )

    out = core.Tensor(backends.get(backend), x * y.data)

    return WrappedFunctionData(wrappedfun, fun, out, (x, y.tensor))




def binary_wrappedfun_dat_003(backend):

    def fun(x, y):

        def _fun(backend, u, v):
            return u * v

        return function.Args(x, y).pluginto(
                  function.TransformCall(_fun)
               )

    wrappedfun = op.typecast_binary(fun)

    x = data.tensor_dat(data.randn)(
           backend, (2,3,4), dtype="complex128", seed=1
        )

    np.random.seed(2)
    y = np.random.randn(1)

    out = core.Tensor(backends.get(backend), x.data * y)

    return WrappedFunctionData(wrappedfun, fun, out, (x.tensor, y))




def binary_wrappedfun_dat_004(backend):

    def fun(x, y):

        def _fun(backend, u, v):
            return u * v

        return function.Args(x, y).pluginto(
                  function.TransformCall(_fun)
               )

    wrappedfun = op.typecast_binary(fun)

    x = data.tensor_dat(data.randn)(
           backend, (2,3,4), dtype="complex128", seed=1
        )
    y = data.tensor_dat(data.randn)(
           backend, (2,3,4), dtype="complex128", seed=2
        )

    out = core.Tensor(backends.get(backend), x.data * y.data)

    return WrappedFunctionData(wrappedfun, fun, out, (x.tensor, y.tensor))




