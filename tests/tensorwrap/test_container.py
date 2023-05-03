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

import tadpole.tensorwrap.container as tc

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

from tadpole.tensorwrap.container import (
   NullGrad,
   SparseGrad,
   ContainerGen,
   ContainerSpace,
)



"""
###############################################################################
###                                                                         ###
###  Special container types for gradients                                  ###
###                                                                         ###
###############################################################################


# --- Sparse gradient ------------------------------------------------------- #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestSparseGrad:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   # --- Container factories --- #

   @pytest.mark.parametrize("size", [0,1,2,3])
   def test_sparsegrad(self, size):

       out = tc.zeros(size)
       ans = tuple(tn.NullGrad() for _ in range(size))
       ans = ContainerGen(ans)

       assert out == ans


   # --- Grad methods --- #

   @pytest.mark.parametrize("shapes, inds, pos", [
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [1,  ]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [2, 0]],
   ]) 
   def test_todense(self, shapes, inds):

       w = data.ntensor_dat(data.randn)(
              self.backend, inds, shapes
           )

       x = SparseGrad(len(shapes), pos, w.tensors)
       y = ContainerGen()

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

       assert x.tonull() == tc.zeros(len(shapes))


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

"""



"""
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

       out = tc.zeros(size)
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

       assert x.tonull() == tc.zeros(len(shapes))


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

       out = tc.addgrads(x,  y)
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

       out = tc.addgrads(x, y)
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

       out = tc.addgrads(y, x)
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

       out = tc.addgrads(x, y)
       ans = list(x.todense())

       for i, p in enumerate(ypos):
           ans[p] = tn.addgrads(ans[p], ytensors[i])

       for outi, ansi in zip(out, ans):
           assert tn.allclose(outi, ansi)


   @pytest.mark.parametrize("shapes, inds", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"]],
   ]) 
   def test_addgrads_null_dense(self, shapes, inds):

       w = data.ntensor_dat(data.randn)(
              self.backend, inds, shapes
           )

       x = NullGrad(len(shapes))
       y = ContainerGen(w.tensors)

       out = tc.addgrads(x, y)
       ans = list(w.tensors)

       for outi, ansi in zip(out, ans):
           assert tn.allclose(outi, ansi)


   @pytest.mark.parametrize("shapes, inds", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"]],
   ]) 
   def test_addgrads_dense_null(self, shapes, inds):

       w = data.ntensor_dat(data.randn)(
              self.backend, inds, shapes
           )

       x = NullGrad(len(shapes))
       y = ContainerGen(w.tensors)

       out = tc.addgrads(y, x)
       ans = list(w.tensors)

       for outi, ansi in zip(out, ans):
           assert tn.allclose(outi, ansi)


   @pytest.mark.parametrize("shapes, inds", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"]],
   ]) 
   def test_addgrads_null_null(self, shapes, inds):

       w = data.ntensor_dat(data.randn)(
              self.backend, inds, shapes
           )

       x = NullGrad(len(shapes))
       y = NullGrad(len(shapes))

       out = tc.addgrads(x, y)
       assert out == x


   @pytest.mark.parametrize("shapes, inds, pos", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [1,  ]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [2, 0]],
   ]) 
   def test_addgrads_null_sparse(self, shapes, inds, pos):

       w = data.ntensor_dat(data.randn)(
              self.backend, inds, shapes
           )

       ytensors = [w.tensors[p] for p in pos]

       x = NullGrad(len(shapes))
       y = SparseGrad(len(shapes), pos, ytensors)

       out = tc.addgrads(x, y)
       ans = y.todense()

       for outi, ansi in zip(out, ans):
           assert tn.allclose(outi, ansi)


   @pytest.mark.parametrize("shapes, inds, pos", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [1,  ]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [2, 0]],
   ]) 
   def test_addgrads_sparse_null(self, shapes, inds, pos):

       w = data.ntensor_dat(data.randn)(
              self.backend, inds, shapes
           )

       ytensors = [w.tensors[p] for p in pos]

       x = NullGrad(len(shapes))
       y = SparseGrad(len(shapes), pos, ytensors)

       out = tc.addgrads(y, x)
       ans = y.todense()

       for outi, ansi in zip(out, ans):
           assert tn.allclose(outi, ansi)














###############################################################################
###                                                                         ###
###  Container grads                                                        ###
###                                                                         ###
###############################################################################


# --- Container grads ------------------------------------------------------- #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestGradsContainer:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend
"""

"""
   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]], 
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_getitem(self, shapes, inds):

       def fun(x, pos):
           return x[pos]

       w = data.ntensor_dat(data.randn)(
              self.backend, inds, shapes
           )

       x = ContainerGen(w.tensors)

       assert_grad(fun, modes="vjp", submode="container", order=2)(x, 0)
       assert False




   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]], 
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_sparsegrad(self, shapes, inds):

       def fun(x, pos, size):
           return tc.sparsegrad(x, pos, size)

       w = data.ntensor_dat(data.randn)(
              self.backend, inds, shapes
           )

       source = ContainerGen(w.tensors)
       x      = source[0]

       assert_grad(fun, modes="vjp", submode="container", order=1)(x, 0, len(source))

"""




###############################################################################
###                                                                         ###
###  Special container types for gradients                                  ###
###                                                                         ###
###############################################################################


# --- Null gradient --------------------------------------------------------- #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestNullGrad:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   # --- Grad methods --- #

   @pytest.mark.parametrize("shapes, inds", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"]],
   ]) 
   def test_todense(self, shapes, inds):

       w = data.null_container_dat(data.randn)(
              self.backend, inds, shapes
           )

       assert w.container.todense() == w.dense


   @pytest.mark.parametrize("shapes, inds", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"]],
   ]) 
   def test_tonull(self, shapes, inds):

       w = data.null_container_dat(data.randn)(
              self.backend, inds, shapes
           )

       assert w.container.tonull() == w.container


   # --- Container methods --- #

   @pytest.mark.parametrize("shapes, inds", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"]],
   ]) 
   def test_copy(self, shapes, inds):

       w = data.null_container_dat(data.randn)(
              self.backend, inds, shapes
           )

       out = w.container.copy()
      
       assert out == w.container
       assert out is not w.container


   @pytest.mark.parametrize("shapes, inds", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"]],
   ]) 
   def test_withdata(self, shapes, inds):

       w = data.null_container_dat(data.randn)(
              self.backend, inds, shapes, seed=10
           )
       v = data.null_container_dat(data.randn)(
              self.backend, inds, shapes, seed=20
           )

       out = w.container.withdata(v.arrays)
       ans = ContainerGen([
                tn.TensorGen(v.arrays[i], w.inds.map(*inds[i])) 
                for i in range(len(shapes))
             ])

       assert out == ans


   @pytest.mark.parametrize("shapes, inds", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"]],
   ]) 
   def test_space(self, shapes, inds):

       w = data.null_container_dat(data.randn)(
              self.backend, inds, shapes
           )

       assert w.container.space() == w.space


   @pytest.mark.parametrize("shapes, inds", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"]],
   ]) 
   def test_item(self, shapes, inds):

       w = data.null_container_dat(data.randn)(
              self.backend, inds, shapes
           )

       for i in range(len(w.tensors)):
           assert w.container.item(i) == w.tensorspaces[i].zeros()


   @pytest.mark.parametrize("shapes, inds", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"]],
   ]) 
   def test_len(self, shapes, inds):

       w = data.null_container_dat(data.randn)(
              self.backend, inds, shapes
           )

       assert len(w.container) == len(w.tensors)


   @pytest.mark.parametrize("shapes, inds", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"]],
   ]) 
   def test_contains(self, shapes, inds):

       w = data.null_container_dat(data.randn)(
              self.backend, inds, shapes
           )

       for i in range(len(w.tensors)):
           assert w.tensors[i] not in w.container 


   @pytest.mark.parametrize("shapes, inds", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"]],
   ]) 
   def test_getitem(self, shapes, inds):

       w = data.null_container_dat(data.randn)(
              self.backend, inds, shapes
           )

       for i in range(len(w.tensors)):
           assert w.container[i] == w.tensorspaces[i].zeros() 


   @pytest.mark.parametrize("shapes, inds", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"]],
   ]) 
   def test_iter(self, shapes, inds):

       w = data.null_container_dat(data.randn)(
              self.backend, inds, shapes
           )

       assert tuple(w.container) == tuple(w.dense)




# --- Sparse gradient ------------------------------------------------------- #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestSparseGrad:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   # --- Grad methods --- #

   @pytest.mark.parametrize("shapes, inds, pos", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [1,  ]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [2, 0]],
   ]) 
   def test_todense(self, shapes, inds, pos):

       w = data.sparse_container_dat(data.randn)(
              self.backend, inds, shapes, pos
           )

       assert w.container.todense() == w.dense


   @pytest.mark.parametrize("shapes, inds, pos", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [1,  ]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [2, 0]],
   ]) 
   def test_tonull(self, shapes, inds, pos):

       w = data.sparse_container_dat(data.randn)(
              self.backend, inds, shapes, pos
           )

       assert w.container.tonull() == NullGrad(w.space)


   # --- Container methods --- #

   @pytest.mark.parametrize("shapes, inds, pos", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [1,  ]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [2, 0]],
   ]) 
   def test_copy(self, shapes, inds, pos):

       w = data.sparse_container_dat(data.randn)(
              self.backend, inds, shapes, pos
           )

       out = w.container.copy()
      
       assert out == w.container
       assert out is not w.container


   @pytest.mark.parametrize("shapes, inds, pos", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [1,  ]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [2, 0]],
   ]) 
   def test_withdata(self, shapes, inds, pos):

       w = data.sparse_container_dat(data.randn)(
              self.backend, inds, shapes, pos, seed=10
           )
       v = data.sparse_container_dat(data.randn)(
              self.backend, inds, shapes, pos, seed=20
           )

       out = w.container.withdata(v.arrays)
       ans = ContainerGen([
                tn.TensorGen(v.arrays[i], w.inds.map(*inds[i])) 
                for i in range(len(shapes))
             ])

       assert out == ans


   @pytest.mark.parametrize("shapes, inds, pos", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [1,  ]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [2, 0]],
   ]) 
   def test_space(self, shapes, inds, pos):

       w = data.sparse_container_dat(data.randn)(
              self.backend, inds, shapes, pos
           )

       assert w.container.space() == w.space


   @pytest.mark.parametrize("shapes, inds, pos", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [1,  ]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [2, 0]],
   ]) 
   def test_item(self, shapes, inds, pos):

       w = data.sparse_container_dat(data.randn)(
              self.backend, inds, shapes, pos
           )

       for i in range(len(w.tensors)):
           if   i in pos:
                assert w.container.item(i) is w.tensors[i]
           else:
                assert w.container.item(i) == w.tensorspaces[i].zeros()


   @pytest.mark.parametrize("shapes, inds, pos", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [1,  ]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [2, 0]],
   ]) 
   def test_len(self, shapes, inds, pos):

       w = data.sparse_container_dat(data.randn)(
              self.backend, inds, shapes, pos
           )

       assert len(w.container) == len(w.tensors)


   @pytest.mark.parametrize("shapes, inds, pos", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [1,  ]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [2, 0]],
   ]) 
   def test_contains(self, shapes, inds, pos):

       w = data.sparse_container_dat(data.randn)(
              self.backend, inds, shapes, pos
           )

       for i in range(len(w.tensors)):
           if   i in pos:
                assert w.tensors[i] in w.container 
           else:
                assert w.tensors[i] not in w.container 


   @pytest.mark.parametrize("shapes, inds, pos", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [1,  ]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [2, 0]],
   ]) 
   def test_getitem(self, shapes, inds, pos):

       w = data.sparse_container_dat(data.randn)(
              self.backend, inds, shapes, pos
           )

       for i in range(len(w.tensors)):
           if   i in pos:
                assert w.container[i] is w.tensors[i]
           else:
                assert w.container[i] == w.tensorspaces[i].zeros() 


   @pytest.mark.parametrize("shapes, inds, pos", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [1,  ]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [2, 0]],
   ]) 
   def test_iter(self, shapes, inds, pos):

       w = data.sparse_container_dat(data.randn)(
              self.backend, inds, shapes, pos
           )

       assert tuple(w.container) == tuple(w.dense)




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


   # --- Grad methods --- #

   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]], 
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_todense(self, shapes, inds):

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes
           )

       assert w.container.todense() == w.container


   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]], 
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_tonull(self, shapes, inds):

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes
           )

       assert w.container.tonull() == NullGrad(w.space)


   # --- Container methods --- #

   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]], 
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_copy(self, shapes, inds):

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes
           )
 
       out = w.container.copy()
      
       assert out == w.container
       assert out is not w.container


   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]], 
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_withdata(self, shapes, inds):

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes, seed=10
           )
       v = data.container_dat(data.randn)(
              self.backend, inds, shapes, seed=20
           )

       out = w.container.withdata(v.arrays)
       ans = ContainerGen([
                tn.TensorGen(v.arrays[i], w.inds.map(*inds[i])) 
                for i in range(len(shapes))
             ])

       assert out == ans


   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]], 
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_space(self, shapes, inds):

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes
           )

       assert w.container.space() == w.space


   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]], 
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_item(self, shapes, inds):

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes
           )

       for i in range(len(w.tensors)):
           assert w.container.item(i) is w.tensors[i]


   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]], 
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_len(self, shapes, inds):

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes
           )

       assert len(w.container) == len(w.tensors)


   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]], 
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_contains(self, shapes, inds):

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes
           )

       for i in range(len(w.tensors)):
           assert w.tensors[i] in w.container 


   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]], 
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_getitem(self, shapes, inds):

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes
           )

       for i in range(len(w.tensors)):
           assert w.container[i] is w.tensors[i]


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
       x = ContainerGen(w.tensors[:-1])

       for i in range(len(w.tensors) - 1):
           assert x[i] is w.tensors[i]
           assert not (x[i] == w.tensors[i+1])


   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]], 
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_iter(self, shapes, inds):

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes
           )

       assert tuple(w.container) == tuple(w.tensors)




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

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes, seed=10
           )
       v = data.container_dat(data.randn)(
              self.backend, inds, shapes, seed=20
           )

       x = w.container
       y = w.container.withdata(v.arrays)

       out = tc.addgrads(x, y)
       ans = [tn.addgrads(xi, yi) for xi, yi in zip(x, y)]

       for outi, ansi in zip(out, ans):
           assert tn.allclose(outi, ansi)


   @pytest.mark.parametrize("shapes, inds, pos", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [1,  ]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [2, 0]],
   ]) 
   def test_addgrads_sparse_dense(self, shapes, inds, pos):

       w = data.sparse_container_dat(data.randn)(
              self.backend, inds, shapes, pos, seed=10
           )
       v = data.container_dat(data.randn)(
              self.backend, inds, shapes, seed=20
           )

       x = w.container
       y = w.container.withdata(v.arrays)

       out = tc.addgrads(x, y)
       ans = list(y)

       for i, p in enumerate(pos):
           ans[p] = tn.addgrads(w.vals[i], ans[p])

       for outi, ansi in zip(out, ans):
           assert tn.allclose(outi, ansi)


   @pytest.mark.parametrize("shapes, inds, pos", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [1,  ]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [2, 0]],
   ]) 
   def test_addgrads_dense_sparse(self, shapes, inds, pos):

       w = data.sparse_container_dat(data.randn)(
              self.backend, inds, shapes, pos, seed=10
           )
       v = data.container_dat(data.randn)(
              self.backend, inds, shapes, seed=20
           )

       x = w.container
       y = w.container.withdata(v.arrays)

       out = tc.addgrads(y, x)
       ans = list(y)

       for i, p in enumerate(pos):
           ans[p] = tn.addgrads(ans[p], w.vals[i])

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

       w = data.sparse_container_dat(data.randn)(
              self.backend, inds, shapes, xpos, seed=10
           )
       v = data.sparse_container_dat(data.randn)(
              self.backend, inds, shapes, ypos, seed=20
           )

       xvals = w.vals
       yvals = [w.tensorspaces[p].fillwith(v.arrays[p]) for p in ypos]

       x = SparseGrad(w.space, xpos, xvals)
       y = SparseGrad(w.space, ypos, yvals)

       out = tc.addgrads(x, y)
       ans = list(w.dense)

       for i, p in enumerate(ypos):
           ans[p] = tn.addgrads(ans[p], yvals[i])

       for outi, ansi in zip(out, ans):
           assert tn.allclose(outi, ansi)


   @pytest.mark.parametrize("shapes, inds", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"]],
   ]) 
   def test_addgrads_null_dense(self, shapes, inds):

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes
           )

       x = w.container
       y = w.container.tonull()

       out = tc.addgrads(x, y)
       ans = list(w.tensors)

       for outi, ansi in zip(out, ans):
           assert tn.allclose(outi, ansi)


   @pytest.mark.parametrize("shapes, inds", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"]],
   ]) 
   def test_addgrads_dense_null(self, shapes, inds):

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes
           )

       x = w.container
       y = w.container.tonull()

       out = tc.addgrads(y, x)
       ans = list(w.tensors)

       for outi, ansi in zip(out, ans):
           assert tn.allclose(outi, ansi)


   @pytest.mark.parametrize("shapes, inds", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"]],
   ]) 
   def test_addgrads_null_null(self, shapes, inds):

       w = data.null_container_dat(data.randn)(
              self.backend, inds, shapes
           )

       x = NullGrad(w.space)
       y = NullGrad(w.space)

       out = tc.addgrads(x, y)
       assert out == x


   @pytest.mark.parametrize("shapes, inds, pos", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [1,  ]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [2, 0]],
   ]) 
   def test_addgrads_null_sparse(self, shapes, inds, pos):

       w = data.sparse_container_dat(data.randn)(
              self.backend, inds, shapes, pos
           )

       x = w.container
       y = w.container.tonull()

       out = tc.addgrads(y, x)
       ans = x.todense()

       for outi, ansi in zip(out, ans):
           assert tn.allclose(outi, ansi)


   @pytest.mark.parametrize("shapes, inds, pos", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [1,  ]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [2, 0]],
   ]) 
   def test_addgrads_sparse_null(self, shapes, inds, pos):

       w = data.sparse_container_dat(data.randn)(
              self.backend, inds, shapes, pos
           )

       x = w.container
       y = w.container.tonull()

       out = tc.addgrads(x, y)
       ans = x.todense()

       for outi, ansi in zip(out, ans):
           assert tn.allclose(outi, ansi)




