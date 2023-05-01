#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import collections
import itertools
import numpy as np

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.tensor   as tn
import tadpole.index    as tid

import tests.tensorwrap.fakes as fake
import tests.tensorwrap.data  as data

import tadpole.tensorwrap.container as tnc


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

from tadpole.tensorwrap.container import (
   NullGrad,
   SparseGrad,
   ContainerGen,
)



###############################################################################
###                                                                         ###
###  General container                                                      ###
###                                                                         ###
###############################################################################


# --- General container ----------------------------------------------------- #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestContainerGen:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   # --- Container factories --- #

   @pytest.mark.parametrize("size", [0,1,2,3])
   def test_zeros(self, size):

       out = tnc.zeros(size)
       ans = tuple(tn.NullGrad() for _ in range(size))
       ans = ContainerGen(ans)

       assert out == ans


   # --- Grad methods --- #

   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]], 
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_todense(self, shapes, inds):

       w = data.ntensor_dat(data.randn)(
              self.backend, inds, shapes
           )

       x = ContainerGen(tuple(w.tensors))

       assert x.todense() == x


   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]], 
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_tonull(self, shapes, inds):

       w = data.ntensor_dat(data.randn)(
              self.backend, inds, shapes
           )

       x = ContainerGen(tuple(w.tensors))

       assert x.tonull() == tnc.zeros(len(shapes))


   # --- Element access --- #

   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]], 
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_item(self, shapes, inds):

       w = data.ntensor_dat(data.randn)(
              self.backend, inds, shapes
           )

       x = ContainerGen(tuple(w.tensors))

       for i in range(len(shapes)):
           assert x.item(i) is w.tensors[i]


   # --- Container methods --- #

   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]], 
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_len(self, shapes, inds):

       w = data.ntensor_dat(data.randn)(
              self.backend, inds, shapes
           )

       x = ContainerGen(tuple(w.tensors))

       assert len(x) == len(shapes)


   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]], 
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_contains(self, shapes, inds):

       w = data.ntensor_dat(data.randn)(
              self.backend, inds, shapes
           )

       x = ContainerGen(tuple(w.tensors))

       for i in range(len(shapes)):
           assert w.tensors[i] in x 


   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]], 
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_getitem(self, shapes, inds):

       w = data.ntensor_dat(data.randn)(
              self.backend, inds, shapes
           )

       x = ContainerGen(tuple(w.tensors))

       for i in range(len(shapes)):
           assert x[i] is w.tensors[i]


   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]], 
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_getitem_001(self, shapes, inds):

       w = data.ntensor_dat(data.randn)(
              self.backend, inds, shapes
           )
       x = ContainerGen(tuple(w.tensors[:-1]))

       for i in range(len(shapes) - 1):
           assert x[i] is w.tensors[i]
           assert not (x[i] == w.tensors[i+1])


   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]], 
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_iter(self, shapes, inds):

       w = data.ntensor_dat(data.randn)(
              self.backend, inds, shapes
           )

       x = ContainerGen(tuple(w.tensors))

       assert tuple(x) == tuple(w.tensors)




###############################################################################
###                                                                         ###
###  Gradient accumulation                                                  ###
###                                                                         ###
###############################################################################


# --- Gradient accumulation ------------------------------------------------- #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestGradAccum:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",              ]], 
      [[(3,4,6), (6,2,5)           ], ["ijk", "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk", "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"]],
   ]) 
   def test_addgrads(self, shapes, inds):

       w = data.ntensor_dat(data.randn)(
              self.backend, 
              inds   + inds, 
              shapes + shapes
           )

       xtensors = w.tensors[: len(shapes)]
       ytensors = w.tensors[len(shapes) :]

       x = ContainerGen(xtensors)
       y = ContainerGen(ytensors)

       out = tnc.addgrads(x,  y)
       ans = [tn.addgrads(xi, yi) for xi, yi in zip(xtensors, ytensors)]

       for outi, ansi in zip(out, ans):
           assert tn.allclose(outi, ansi)


   @pytest.mark.parametrize("shapes, inds, pos", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [1,  ]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [2, 0]],
   ]) 
   def test_addgrads_sparse_dense(self, shapes, inds, pos):

       w = data.ntensor_dat(data.randn)(
              self.backend, 
              inds   + inds, 
              shapes + shapes
           )

       xtensors = [w.tensors[p] for p in pos]
       ytensors = w.tensors[len(shapes) :]

       x = SparseGrad(len(shapes), pos, xtensors)
       y = ContainerGen(ytensors)

       out = tnc.addgrads(x, y)
       ans = list(ytensors)

       for i, p in enumerate(pos):
           ans[p] = tn.addgrads(xtensors[i], ans[p])

       for outi, ansi in zip(out, ans):
           assert tn.allclose(outi, ansi)


   @pytest.mark.parametrize("shapes, inds, pos", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [1,  ]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [2, 0]],
   ]) 
   def test_addgrads_dense_sparse(self, shapes, inds, pos):

       w = data.ntensor_dat(data.randn)(
              self.backend, 
              inds   + inds, 
              shapes + shapes
           )

       xtensors = [w.tensors[p] for p in pos]
       ytensors = w.tensors[len(shapes) :]

       x = SparseGrad(len(shapes), pos, xtensors)
       y = ContainerGen(ytensors)

       out = tnc.addgrads(y, x)
       ans = list(ytensors)

       for i, p in enumerate(pos):
           ans[p] = tn.addgrads(ans[p], xtensors[i])

       for outi, ansi in zip(out, ans):
           assert tn.allclose(outi, ansi)


   @pytest.mark.parametrize("shapes, inds, xpos, ypos", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [1,  ], [1,  ]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [1,  ], [0,  ]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [2, 0], [1,  ]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [2, 0], [0,  ]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [2, 0], [1, 0]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [0,  ], [2, 0]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [1,  ], [2, 0]],
   ]) 
   def test_addgrads_sparse_sparse(self, shapes, inds, xpos, ypos):

       w = data.ntensor_dat(data.randn)(
              self.backend, 
              inds   + inds, 
              shapes + shapes
           )

       xtensors = [w.tensors[p] for p in xpos]
       ytensors = [w.tensors[p] for p in ypos]

       x = SparseGrad(len(shapes), xpos, xtensors)
       y = SparseGrad(len(shapes), ypos, ytensors)

       out = tnc.addgrads(x, y)
       ans = list(x.todense())

       for i, p in enumerate(ypos):
           ans[p] = tn.addgrads(ans[p], ytensors[i])

       for outi, ansi in zip(out, ans):
           assert tn.allclose(outi, ansi)












