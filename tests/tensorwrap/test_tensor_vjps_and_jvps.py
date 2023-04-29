#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import collections
import itertools
import numpy as np

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.array    as ar
import tadpole.tensor   as tn
import tadpole.index    as tid

import tadpole.array.backends as backends

import tadpole.tensorwrap.tensor_vjps as tvjps
import tadpole.tensorwrap.tensor_jvps as tjvps

import tests.tensorwrap.fakes as fake
import tests.tensorwrap.data  as data
import tests.array.data       as ardata


from tests.common import (
   available_backends,
)

from tests.tensorwrap.util import (
   assert_grad,
)

from tadpole.index import (
   Index,
   IndexGen,  
   Indices,
)

from tadpole.tensorwrap.tensor_vjps.elemwise_unary import (
   sparsegrad,
)




###############################################################################
###                                                                         ###
###  Unary elementwise grads                                                ###
###                                                                         ###
###############################################################################


# --- Unary elementwise grads ----------------------------------------------- #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestGradsElemwiseUnary:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   @pytest.mark.parametrize("indnames, shape", [
      ["ijk", (2,3,4)],
   ])
   @pytest.mark.parametrize("op", [
      "neg",
      "conj",
      "real",
      "imag",
      "absolute",
      "sqrt",
      "log",
      "exp",
      "sin", 
      "cos",
      "tan",
      "arcsin",
      "arccos",
      "arctan",
      "sinh", 
      "cosh",
      "tanh",
      "arcsinh",
      "arccosh",
      "arctanh",
   ])
   def test_math(self, indnames, shape, op):

       opts = {}
       fun = {
              "neg":      lambda x: -x,
              "conj":     lambda x: tn.conj(x),
              "real":     lambda x: tn.real(x),
              "imag":     lambda x: tn.imag(x),
              "absolute": lambda x: tn.absolute(x),
              "sqrt":     lambda x: tn.sqrt(x),
              "log":      lambda x: tn.log(x),
              "exp":      lambda x: tn.exp(x),  
              "sin":      lambda x: tn.sin(x),
              "cos":      lambda x: tn.cos(x),
              "tan":      lambda x: tn.tan(x),
              "arcsin":   lambda x: tn.arcsin(x),
              "arccos":   lambda x: tn.arccos(x),
              "arctan":   lambda x: tn.arctan(x),
              "sinh":     lambda x: tn.sinh(x), 
              "cosh":     lambda x: tn.cosh(x),
              "tanh":     lambda x: tn.tanh(x),
              "arcsinh":  lambda x: tn.arcsinh(x),
              "arccosh":  lambda x: tn.arccosh(x),
              "arctanh":  lambda x: tn.arctanh(x),
             }[op]

       x = data.tensor_dat(data.randn)(
              self.backend, indnames, shape, seed=1
           )
       xtensor = x.tensor
 
       if op == "arccosh":
          xtensor = xtensor + 2.5

       if op in ("conj", "real", "imag"): 
          opts = {"submode": "real"}

       if op in ("absolute",):
          opts = {"order": 3, "submode": "real"}

       assert_grad(fun, **opts)(xtensor)

   
   @pytest.mark.parametrize("indnames, shape, minval, maxval", [
      ["ijk", (2,3,4), 0, 1], 
   ])
   def test_clip(self, indnames, shape, minval, maxval):

       def fun(x, minval, maxval):
           return tn.clip(x, minval, maxval)

       x = data.tensor_dat(data.randn)(
              self.backend, indnames, shape, seed=1
           )

       assert_grad(fun)(x.tensor, minval, maxval)


   @pytest.mark.parametrize("indnames, shape, inds", [
      ["ijk", (2,3,4), None], 
      ["ijk", (2,3,4), "i"],
      ["ijk", (2,3,4), "ki"],
      ["ijk", (2,3,4), "kij"],
   ])
   def test_flip(self, indnames, shape, inds):

       def fun(x, inds):
           return tn.flip(x, inds)

       x = data.tensor_dat(data.randn)(
              self.backend, indnames, shape, seed=1
           )

       assert_grad(fun)(x.tensor, inds)


   @pytest.mark.parametrize("indnames, shape, ind", [
      ["ijk", (2,3,4), None],
      ["ijk", (2,3,4), "j"],
   ])
   def test_cumsum(self, indnames, shape, ind):

       def fun(x, ind):
           return tn.cumsum(x, ind)

       x = data.tensor_dat(data.randn)(
              self.backend, indnames, shape, seed=1
           )

       assert_grad(fun)(x.tensor, ind)


   @pytest.mark.parametrize("indnames, shape", [
      ["ijk", (2,3,4)],
   ])
   def test_getitem(self, indnames, shape):

       def fun(x, pos):
           return x[pos]

       x = data.tensor_dat(data.randn)(
              self.backend, indnames, shape, seed=1
           )
       pos = (1,0,2)

       """
       pos = (
              ((1,0,1),), 
              ((0,2,0),),
              ((2,1,3),),
             )
       """

       """
       pos = (
              ((1,),), 
              ((0,),),
              ((2,),),
             )
       """

       assert_grad(fun)(x.tensor, pos)


   @pytest.mark.parametrize("graddat", [
      data.sparsegrad_dat_001,
   ])
   @pytest.mark.parametrize("dtype", [
      "complex128",
   ])
   def test_sparsegrad(self, graddat, dtype):

       def fun(x, pos, source):
           return sparsegrad(x, pos, source)

       w = graddat(
              self.backend, dtype, seed=1
           )   

       x   = tn.astensor(w.vals[0])
       pos = w.pos[0]

       assert_grad(fun)(x, pos, tn.space(w.tensor))




###############################################################################
###                                                                         ###
###  Binary elementwise grads                                               ###
###                                                                         ###
###############################################################################


# --- Binary elementwise grads ---------------------------------------------- #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestGradsElemwiseBinary:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   @pytest.mark.parametrize("indnames, shape", [
      ["ijk", (2,3,4)],
   ])
   @pytest.mark.parametrize("op", [
      "add", 
      "sub", 
      "mul", 
      "div", 
      "pow",
      "addgrads",
   ])
   def test_math(self, indnames, shape, op):

       fun = {
              "add":      lambda x, y: x + y,
              "sub":      lambda x, y: x - y,
              "mul":      lambda x, y: x * y,
              "div":      lambda x, y: x / y,
              "pow":      lambda x, y: x ** y,
              "addgrads": lambda x, y: tn.addgrads(x, y),
             }[op]

       x = data.tensor_dat(data.randn)(
              self.backend, indnames, shape, seed=1
           )
       y = data.tensor_dat(data.randn)(
              self.backend, indnames, shape, seed=2
           )

       xtensor = x.tensor
       ytensor = tn.TensorGen(y.array, x.inds)

       assert_grad(fun, 0)(xtensor, ytensor)
       assert_grad(fun, 1)(xtensor, ytensor)


   @pytest.mark.parametrize("sampledat", [
      ardata.randuniform_real_dat_001,
   ])
   @pytest.mark.parametrize("op", [
      "mod", 
   ])
   def test_math_int(self, sampledat, op):

       fun = {
              "mod": lambda x, y: x % y,
             }[op]

       x = data.tensor_sample_dat(sampledat)(
              self.backend, boundaries=(1,11), seed=1
           )
       y = data.tensor_sample_dat(sampledat)(
              self.backend, boundaries=(1,11), seed=2
           )

       xtensor = x.tensor
       ytensor = tn.TensorGen(y.array, x.inds)

       assert_grad(fun, 0)(xtensor, ytensor)
       assert_grad(fun, 1)(xtensor, ytensor)




###############################################################################
###                                                                         ###
###  Ternary elementwise grads                                              ###
###                                                                         ###
###############################################################################


# --- Ternary elementwise grads --------------------------------------------- #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestGradsElemwiseTernary:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   @pytest.mark.parametrize("indnames, shape, nvals", [
      ["ijk", (2,3,4), 5],
   ])
   def test_where(self, indnames, shape, nvals):

       w = data.tensor_dat(data.randn_pos)(
              self.backend, 
              indnames, 
              shape, 
              seed=1, 
              dtype="bool", 
              nvals=nvals, 
              defaultval=False
           )
       x = data.array_dat(data.randn)(
              self.backend, w.shape, seed=2
           )
       y = data.array_dat(data.randn)(
              self.backend, w.shape, seed=3
           )

       wtensor = w.tensor
       xtensor = tn.TensorGen(x.array, w.inds)
       ytensor = tn.TensorGen(y.array, w.inds)

       def fun(w,x,y):
           return tn.where(w,x,y)

       assert_grad(fun, 0, submode="null")(wtensor, xtensor, ytensor)
       assert_grad(fun, 1                )(wtensor, xtensor, ytensor)
       assert_grad(fun, 2                )(wtensor, xtensor, ytensor)










"""
   @pytest.mark.parametrize("shape, inds, diffinds", [
      [(2,3,4), "ijk", "k"],  
      [(2,3,4), "ijk", None],     
   ])
   def test_unreduce_like(self, shape, inds, diffinds): # TODO move to tensorwrap tests (test vjp/jvp_reduce)

       w = data.tensor_dat(data.randn)(
              self.backend, inds, shape
           )

       target = w.tensor
       x      = tn.amax(target, diffinds)

       if   diffinds is None:
            fun = tn.unreduce_like(x, target)
       else:
            fun = tn.unreduce_like(x, target, w.inds.map(*diffinds))

       out    = fun(x)
       outmax = tn.amax(out, diffinds)

       assert out.space() == target.space()
       assert tn.allclose(outmax, x)
"""
       



