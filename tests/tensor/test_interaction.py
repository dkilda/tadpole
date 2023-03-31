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
import tadpole.tensor.interaction as tni
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
###  Tensor interaction engine and operator                                 ###
###                                                                         ###
###############################################################################


# --- Tensor interaction operator ------------------------------------------- # 

@pytest.mark.parametrize("current_backend", ["numpy_backend"], indirect=True)
class TestTensorInteraction:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   # --- Mutual index methods --- #

   @pytest.mark.parametrize("shapes, inds, output", [
      [[(3,4,5),                   ], ["ijk",              ], "ijk"   ],  
      [[(3,4,6), (6,3,4)           ], ["ijk", "kij",       ], "ijk"   ], 
      [[(3,4,6), (6,2,5)           ], ["ijk", "klm",       ], "ijklm" ],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], "ijklmq"], 
   ])
   def test_union_inds(self, shapes, inds, output):

       w = data.ntensor_dat(data.randn)(
              self.backend, inds, shapes
           )

       out = tuple(tn.union_inds(*w.tensors))
       ans = w.inds.map(*output)

       assert out == ans


   @pytest.mark.parametrize("shapes, inds, output", [
      [[(3,4,5),                   ], ["ijk",              ], "ijk" ],  
      [[(3,4,6), (6,3,4)           ], ["ijk", "kij",       ], "ijk" ], 
      [[(3,4,6), (6,2,5)           ], ["ijk", "klm",       ],  "k"  ],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"],  ""   ], 
      [[(5,3,2), (6,2,5), (5,7,2,4)], ["mil", "klm", "mqlj"],  "ml" ], 
   ])
   def test_overlap_inds(self, shapes, inds, output):

       w = data.ntensor_dat(data.randn)(
              self.backend, inds, shapes
           )

       out = tuple(tn.overlap_inds(*w.tensors))
       ans = w.inds.map(*output)

       assert out == ans


   @pytest.mark.parametrize("shapes, inds, output", [
      [[(3,4,5),                       ], ["ijk",                ], "ijk"],  
      [[(3,4,6),     (6,3,4)           ], ["ijk",   "kij",       ], ""   ], 
      [[(3,4,6),     (6,2,5)           ], ["ijk",   "klm",       ], "ij" ],  
      [[(6,2,5),     (3,4,6),          ], ["klm",   "ijk",       ], "lm" ], 
      [[(3,4,6),     (6,2,5), (5,7,2,4)], ["ijk",   "klm", "mqlj"], "i"  ], 
      [[(5,8,7,2,4), (3,4,6), (6,2,5)  ], ["mpqlj", "ijk", "klm" ], "pq" ], 
   ])
   def test_complement_inds(self, shapes, inds, output):

       w = data.ntensor_dat(data.randn)(
              self.backend, inds, shapes
           )

       out = tuple(tn.complement_inds(*w.tensors))
       ans = w.inds.map(*output)

       assert out == ans


   # --- Tensor matching --- #

   @pytest.mark.filterwarnings('ignore::RuntimeWarning')
   @pytest.mark.parametrize("dtypes, outdtype", [
      [["float64",    "float64"   ], "float64"   ],
      [["complex128", "complex128"], "complex128"],
      [["float64",    "complex128"], "complex128"],
      [["complex128", "float64"   ], "float64"   ],
   ])
   def test_astype_like(self, dtypes, outdtype):

       inds  = "ijk"
       shape = (2,3,4)

       x = data.tensor_dat(data.randn)(
              self.backend, inds, shape, dtype=dtypes[0], seed=1
           )
       y = data.tensor_dat(data.randn)(
              self.backend, inds, shape, dtype=dtypes[1], seed=2
           )

       out = tn.astype_like(x.tensor, y.tensor)
       ans = tn.TensorGen(x.array, x.inds)
       ans = tn.astype(ans, dtypes[1])

       assert out.dtype == outdtype
       assert out       == ans


   @pytest.mark.parametrize("shapes, inds, outinds, keepinds", [
      [[(3,4,5),   (3,4,5)      ], ["ijk",  "ijk"   ], "ijk",    None], 
      [[(3,4,5),   (3,1,4,5)    ], ["ijk",  "imjk"  ], "ijk",    None],
      [[(3,4,5),   (3,1,4,5)    ], ["ijk",  "imjk"  ], "ijk",    False],
      [[(3,4,5),   (3,1,4,5)    ], ["ijk",  "imjk"  ], "imjk",   True],
      [[(3,4,5),   (3,2,4,5)    ], ["ijk",  "imjk"  ], "imjk",   None], 
      [[(3,2,4,5), (3,4,5)      ], ["imjk", "ijk"   ], "ijk",    None],
      [[(3,4,5),   (3,2,1,4,1,5)], ["ijk",  "imnjpk"], "imjk",   None],
      [[(3,4,5),   (3,2,1,4,1,5)], ["ijk",  "imnjpk"], "imjk",   False],
      [[(3,4,5),   (3,2,1,4,1,5)], ["ijk",  "imnjpk"], "imnjpk", True],
   ])
   def test_reshape_like(self, shapes, inds, outinds, keepinds):

       w = data.ntensor_dat(data.randn)(
              self.backend, inds, shapes
           )

       if  keepinds is None:
           out = tn.reshape_like(w.tensors[0], w.tensors[1])
       else:
           out = tn.reshape_like(w.tensors[0], w.tensors[1], keepinds=keepinds)

       assert tuple(tn.union_inds(out)) == w.inds.map(*outinds)


   @pytest.mark.parametrize("shape, inds, diffinds", [
      [(2,3,4), "ijk", "k"],  
      [(2,3,4), "ijk", None],     
   ])
   def test_unreduce_like(self, shape, inds, diffinds):

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
 
       





