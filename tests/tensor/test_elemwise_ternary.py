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

import tadpole.array.backends          as backends
import tadpole.tensor.elemwise_ternary as tnt
import tadpole.tensor.engine           as tne 

import tests.tensor.fakes as fake
import tests.tensor.data  as data


from tests.common import (
   available_backends,
)


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
###  Tensor ternary elementwise engine and operator                         ###
###                                                                         ###
###############################################################################


# --- Tensor ternary elementwise operator ----------------------------------- #  

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestTensorElemwiseTernary:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   # --- Value methods --- #

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

       out = tn.where(wtensor, xtensor, ytensor)
       ans = ar.where(w.array, x.array, y.array)

       assert tn.allclose(out, ans)















