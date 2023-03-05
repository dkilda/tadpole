#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import numpy as np

import tests.array.fakes as fake
import tests.array.data  as data

import tadpole.util           as util
import tadpole.array.core     as core
import tadpole.array.grad     as grad
import tadpole.array.backends as backends
import tadpole.array.function as function




# --- Function data ------------------------------------------------------------ #

FunctionData = collections.namedtuple("FunctionData", [
                  "funcall", "content", "args", 
                  "fun", "out", 
                  "arrays", "datas", "arrays_and_datas",
               ])




def _function_dat(backend, shape, dtype, nargs, which="function"):

    xs = []
    for i in range(nargs):

        x = data.array_dat(data.randn)(
               backend, shape, dtype=dtype, seed=i+1
            )
        xs.append(x)
     
    arrays = [x.array for x in xs]
    datas  = [x.data  for x in xs]
    seq    = list(zip(arrays, datas))

    if   which == "function":
 
         out = data.array_dat(data.randn)(
                  backend, shape, dtype=dtype, seed=nargs+1
               ) 
         fun = fake.Fun(out.data, out.backend, *datas)
         out = out.array

         funcall = function.FunCall(fun, util.Sequence(seq))

    elif which == "visit":

         out = fake.Value()
         fun = fake.Fun(out, xs[0].backend, *datas)

         funcall = function.Visit(fun, util.Sequence(seq))

    else:
         raise ValueError(f"_function_dat: invalid input which {which}") 

    content = function.Content(util.Sequence(seq))
    args    = function.Args(*arrays)

    return FunctionData(
                        funcall, content, args, 
                        fun, 
                        util.Outputs(out),
                        arrays, datas, seq,
                       )




def function_dat(backend, nargs):

    return _function_dat(backend, (2,3,4), "complex128", nargs, "function")




def visit_dat(backend, nargs):

    return _function_dat(backend, (2,3,4), "complex128", nargs, "visit")



































