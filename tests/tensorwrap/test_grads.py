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




###############################################################################
###                                                                         ###
###  Binary elementwise grads                                               ###
###                                                                         ###
###############################################################################


# --- Binary elementwise grad ----------------------------------------------- #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestVjpElemwiseBinary:

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
   ])
   def test_arithmetics(self, indnames, shape, op):

       fun = {
              "add": lambda x, y: x + y,
              "sub": lambda x, y: x - y,
              "mul": lambda x, y: x * y,
              "div": lambda x, y: x / y,
              "pow": lambda x, y: x ** y,
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
       



