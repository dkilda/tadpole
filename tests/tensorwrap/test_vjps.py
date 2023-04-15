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
###  Binary elementwise VJPs                                                ###
###                                                                         ###
###############################################################################


# --- Binary elementwise VJPs ----------------------------------------------- #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestVjpElemwiseBinary:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   def test_vjp(self): # TODO dummy test, replace it!

       backend = backends.get(self.backend)
       assert backend.name() == self.backend


   @pytest.mark.parametrize("indnames, shape", [
      ["ijk", (2,3,4)],
   ])
   def test_add(self, indnames, shape):

       def fun(x, y):
           return x + y

       x = data.tensor_dat(data.randn)(
              self.backend, indnames, shape, seed=1
           )
       y = data.tensor_dat(data.randn)(
              self.backend, indnames, shape, seed=2
           )

       xtensor = x.tensor
       ytensor = tn.TensorGen(y.array, x.inds)

       assert_grad(fun, 0, order=1)(xtensor, ytensor)   # TODO CHANGE BACK TO order=2 once we implement NodeTuple! 
       assert_grad(fun, 1, order=1)(xtensor, ytensor)   # TODO CHANGE BACK TO order=2 once we implement NodeTuple! 
"""
"""









