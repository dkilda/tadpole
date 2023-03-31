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

import tadpole.array.backends     as backends
import tadpole.tensor.contraction as tnc
import tadpole.tensor.engine      as tne 

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
###  Tensor contraction operator                                            ###
###  (includes contract, dot, and other operations)                         ###
###                                                                         ###
###############################################################################


# --- Tensor contraction operator ------------------------------------------- #

@pytest.mark.parametrize("current_backend", ["numpy_backend"], indirect=True)
class TestTensorContract:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   # --- Contraction --- #

   @pytest.mark.parametrize("shapes, inds, outinds, equation", [
      [[(3,4,4),                   ], ["ijj",              ], "i",    "ijj->i"          ],  
      [[(4,4),                     ], ["jj",               ], "",     "jj->"            ],
      [[(3,4,6), (6,3,4)           ], ["ijk", "kij",       ], "",     "ijk,kij->"       ], 
      [[(3,4,6), (6,2,5)           ], ["ijk", "klm",       ], "ijlm", "ijk,klm->ijlm"   ],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], "iq",   "ijk,klm,mqlj->iq"], 
   ])   
   def test_contract(self, inds, shapes, outinds, equation):

       w = data.ntensor_dat(data.randn)(
              self.backend, inds, shapes
           )

       out = tn.contract(*w.tensors)
       ans = ar.einsum(equation, *w.arrays)
       ans = tn.TensorGen(ans,    w.inds.map(*outinds))

       assert tn.allclose(out, ans)
       assert out.space() == ans.space()

      
   @pytest.mark.parametrize("shapes, inds, outinds, equation", [
      [[(3,4,4),                   ], ["ijj",              ], "i",    "ijj->i"           ],  
      [[(4,4,4),                   ], ["jjj",              ], "",     "jjj->"            ],
      [[(3,4,6), (6,3,4)           ], ["ijk", "kij",       ], "",     "ijk,kij->"        ], 
      [[(3,4,6), (6,2,5)           ], ["ijk", "klm",       ], "ijlm", "ijk,klm->ijlm"    ],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], "imq",  "ijk,klm,mqlj->imq"],
   ])   
   def test_contract_fixed(self, inds, shapes, outinds, equation):

       w = data.ntensor_dat(data.randn)(
              self.backend, inds, shapes
           )

       out = tn.contract(*w.tensors, product=outinds)
       ans = ar.einsum(equation, *w.arrays)
       ans = tn.TensorGen(ans,    w.inds.map(*outinds))

       assert tn.allclose(out, ans)
       assert out.space() == ans.space()


   # --- Dot --- #

   @pytest.mark.parametrize("shapes, inds, outinds", [
      [[(3,4),   (4,6)  ], ["ij",  "jk" ], "ik"  ], 
      [[(2,3,4), (5,4,6)], ["mij", "njk"], "mink"],  
      [[(2,3,4), (4,)   ], ["mij", "j"  ], "mi"  ], 
   ])   
   def test_dot(self, inds, shapes, outinds):

       w = data.ntensor_dat(data.randn)(
              self.backend, inds, shapes
           )

       out = tn.dot(*w.tensors)
       ans = ar.dot(*w.arrays)
       ans = tn.TensorGen(ans, w.inds.map(*outinds))

       assert tn.allclose(out, ans)
       assert out.space() == ans.space()


   # --- Kronecker product --- #

   @pytest.mark.parametrize("shapes, inds, outshape, outinds", [ 
      [[(2,5,4), (3,6,7)], ["ijk", "lmn"], (6,30,28), "abc"],  
   ])   
   def test_kron(self, inds, shapes, outshape, outinds):

       v = data.indices_dat(outinds, outshape)
       w = data.ntensor_dat(data.randn)(self.backend, inds, shapes)

       indmap = dict(zip(zip(*inds), v.inds))

       out = tn.kron(*w.tensors, indmap)
       ans = ar.kron(*w.arrays)
       ans = ar.reshape(ans, outshape)
       ans = tn.TensorGen(ans, v.inds.map(*outinds))

       assert tn.allclose(out, ans)
       assert out.space() == ans.space()




