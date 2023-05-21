#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import collections
import itertools
import numpy as np

import tadpole.util      as util
import tadpole.autodiff  as ad
import tadpole.array     as ar
import tadpole.container as tc
import tadpole.tensor    as tn
import tadpole.index     as tid

import tadpole.linalg.unwrapped as la

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
   IndexLit, 
   Indices,
)




###############################################################################
###                                                                         ###
###  Linalg decomposition grads                                             ###
###                                                                         ###
###############################################################################


# --- Linalg decomposition grads -------------------------------------------- #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestGradsDecomp:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   @pytest.mark.skip
   @pytest.mark.parametrize("decomp_input", [
      #data.decomp_input_001,
      data.decomp_input_002,
      #data.decomp_input_003,
   ])
   def test_svd(self, decomp_input):

       def fun(x, sind):
           return la.svd(x, sind)
           #print("FUN RETURN ", out, out._source._source)
           #return out

           #U, S, VH, error = la.svd(x, sind) 

           """
           try:
              print("FUN: ", S, S._inds)
           except AttributeError:
              print("FUN: ", S, S._source._source._inds)
           """

           #return tc.ascontainer(tn.absolute(U), S, tn.absolute(VH)) 
         
       w = data.svd_tensor_dat(decomp_input)(
              data.randn, self.backend, dtype="float64" #"complex128" #"float64"
           )

       lind = IndexGen("l", w.xmatrix.shape[0])
       rind = IndexGen("r", w.xmatrix.shape[1])

       x = tn.TensorGen(w.xmatrix, (lind, rind)) 

       assert_grad(fun, order=2, modes="vjp", submode="decomp")(x, sind="s")     
       #assert False












