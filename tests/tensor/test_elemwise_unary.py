#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import collections
import numpy as np

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.array    as ar
import tadpole.tensor   as tn
import tadpole.index    as tid

import tadpole.tensor.elemwise_unary  as unary
import tadpole.tensor.engine          as tne 

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




###############################################################################
###                                                                         ###
###  Tensor unary elementwise engine and operator                           ###
###                                                                         ###
###############################################################################


# --- Tensor unary elementwise operator ------------------------------------- #  

@pytest.mark.parametrize("current_backend", ["numpy_backend"], indirect=True)
class TestTensorElemwiseUnary:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   # --- Value methods --- #


   # --- Element access --- #


   # --- Extracting info --- #


   # --- Standard math --- #

   @pytest.mark.parametrize("indnames, shape", [
      ["ijk", (2,3,4)],
   ])
   def test_sin(self, indnames, shape):

       w = data.tensor_dat(data.randn)(
              self.backend, indnames, shape
           )

       out = tn.sin(w.tensor)
       ans = ar.sin(w.array)
       ans = tn.TensorGen(ans, w.inds)

       assert tn.allclose(out, ans)




   # --- Linear algebra --- #





























