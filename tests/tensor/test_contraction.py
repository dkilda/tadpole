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
###  The logic of index contraction: figures out the output indices,        ###
###  generates einsum equations from input and output indices.              ###
###                                                                         ###
###############################################################################


# --- Create einsum equation from input and output indices ------------------ #

class TestMakeEquation:

   @pytest.mark.parametrize("shapes, inds, outinds, equation", [
      [[(3,4,4),                   ], ["ijj",              ], "i",    "abb->a"           ],  
      [[(4,4),                     ], ["jj",               ], "",     "aa->"             ],
      [[(4,4,4),                   ], ["jjj",              ], "",     "aaa->"            ],
      [[(3,4,6), (6,3,4)           ], ["ijk", "kij",       ], "",     "abc,cab->"        ], 
      [[(3,4,6), (6,2,5)           ], ["ijk", "klm",       ], "ijlm", "abc,cde->abde"    ],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], "iq",   "abc,cde,efdb->af" ], 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], "imq",  "abc,cde,efdb->aef"],
   ]) 
   def test_make_equation(self, shapes, inds, outinds, equation):

       w = data.nindices_dat(inds, shapes)

       input_inds  = tuple(w.inds.map(*xinds) for xinds in inds)
       output_inds = w.inds.map(*outinds)

       assert tnc.make_equation(input_inds, output_inds) == equation




# --- Index products -------------------------------------------------------- #

class TestIndexProduct:

   @pytest.mark.parametrize("shapes, inds, outinds", [
      [[(3,4,4),                   ], ["ijj",              ], "i",  ],  
      [[(4,4,4),                   ], ["jjj",              ], "",   ],
      [[(3,4,6), (6,3,4)           ], ["ijk", "kij",       ], "",   ],
      [[(3,4,6), (6,2,5)           ], ["ijk", "klm",       ], "ijlm"],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], "imq" ],
   ])
   def test_fixed(self, shapes, inds, outinds):

       w = data.nindices_dat(inds, shapes)

       input_inds  = [w.inds.map(*xinds) for xinds in inds]
       output_inds = w.inds.map(*outinds)

       prod = tnc.FixedIndexProduct(outinds)
       assert tuple(prod(input_inds)) == output_inds

       prod1 = tnc.FixedIndexProduct(output_inds)
       assert tuple(prod1(input_inds)) == output_inds


   @pytest.mark.parametrize("shapes, inds, outinds", [
      [[(3,4,4),                   ], ["ijj",              ], "i",  ],  
      [[(4,4),                     ], ["jj",               ], "",   ],
      [[(3,4,6), (6,3,4)           ], ["ijk", "kij",       ], "",   ],
      [[(3,4,6), (6,2,5)           ], ["ijk", "klm",       ], "ijlm"],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], "iq"  ],
   ])
   def test_pairwise(self, shapes, inds, outinds):

       w = data.nindices_dat(inds, shapes)

       input_inds  = [w.inds.map(*xinds) for xinds in inds]
       output_inds = w.inds.map(*outinds)

       prod = tnc.PairwiseIndexProduct()
       assert tuple(prod(input_inds)) == output_inds




###############################################################################
###                                                                         ###
###  Tensor contraction operator                                            ###
###  (includes contract, dot, and other operations)                         ###
###                                                                         ###
###############################################################################


# --- Tensor contraction operator ------------------------------------------- #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
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
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",   "klm" ], "ijlm", "ijk,,klm->ijlm"  ],  
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




