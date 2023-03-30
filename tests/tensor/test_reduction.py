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

import tadpole.array.backends   as backends
import tadpole.tensor.reduction as redu
import tadpole.tensor.engine    as tne 

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
###  Tensor reduction engine and operator                                   ###
###                                                                         ###
###############################################################################


# --- Tensor reduction operator --------------------------------------------- #  

@pytest.mark.parametrize("current_backend", ["numpy_backend"], indirect=True)
class TestTensorReduce:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   # --- Value methods --- #

   @pytest.mark.parametrize("winds, wshape, inds", [
      ["ijk", (2,3,4), None],
      ["ijk", (2,3,4), "i"],
      ["ijk", (2,3,4), "ki"],
      ["ijk", (2,3,4), "kij"],
   ])
   def test_allof(self, winds, wshape, inds):

       w = data.tensor_dat(data.randn)(
              self.backend, winds, wshape
           )

       if   inds is None:
            out = tn.allof(w.tensor)
            ans = ar.allof(w.array)
            ans = tn.TensorGen(ans)

       else:
            output_inds = "".join(util.complement(winds, inds)) 

            out = tn.allof(w.tensor, inds)
            ans = ar.allof(w.array,  w.inds.axes(*inds))
            ans = tn.TensorGen(ans,  w.inds.map(*output_inds))

       assert tn.allclose(out, ans)






   # --- Shape methods --- #





   # --- Linear algebra methods --- #

























