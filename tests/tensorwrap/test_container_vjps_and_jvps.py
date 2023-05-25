#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import collections
import itertools
import numpy as np

import tadpole.util      as util
import tadpole.autodiff  as ad
import tadpole.container as tc
import tadpole.tensor    as tn
import tadpole.index     as tid

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
   Indices,
)

from tadpole.container import (
   NullGrad,
   SparseGrad,
   ContainerGen,
   ContainerSpace,
)




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


   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]],   
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_container(self, shapes, inds):

       def fun(*xs):
           return tc.container(*xs)

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes
           )

       assert_grad(fun, submode="container")(*w.tensors)


   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]],   
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_getitem(self, shapes, inds):

       positions = {
          1: [0,       slice(0,1),                       ],
          2: [0, 1,    slice(0,1), slice(0,2), slice(1,2)],
          3: [0, 1, 2, slice(0,1), slice(0,2), slice(1,2), 
                       slice(1,3), slice(2,3), slice(0,3)],
       }[len(shapes)]


       def fun(x, pos):
           return x[pos]

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes
           )

       for pos in positions:
           assert_grad(fun, submode="container")(w.container, pos)


   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]],   
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_ungetitem(self, shapes, inds):

       positions = {
          1: [0,       slice(0,1),                       ],
          2: [0, 1,    slice(0,1), slice(0,2), slice(1,2)],
          3: [0, 1, 2, slice(0,1), slice(0,2), slice(1,2), 
                       slice(1,3), slice(2,3), slice(0,3)],
       }[len(shapes)]

       def fun(x, pos, space):
           return tc.ungetitem(x, pos, space)

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes
           )

       for pos in positions:

           x = w.tensors[pos]

           if isinstance(pos, slice):
              x = ContainerGen(x)

           assert_grad(fun, submode="container")(x, pos, w.space)


   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]], 
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_cmap(self, shapes, inds):

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes
           )

       def fun(x, y):
           return tc.cmap(lambda a, b: a + b, x, y)

       assert_grad(fun, 0, submode="container")(w.container, w.container)
       assert_grad(fun, 1, submode="container")(w.container, w.container)


   def test_cmap_001(self):

       inds   = ["ijk",   "klm",   "mqlj"]
       shapes = [(3,4,6), (6,2,5), (5,7,2,4)]

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes
           )

       u = ContainerGen(ContainerGen(w.tensors[0], w.tensors[2]), w.tensors[1])
       v = ContainerGen(ContainerGen(w.tensors[0], w.tensors[2]), w.tensors[1])

       def fun(x, y):
           return tc.cmap(lambda a, b: a + b, x, y)

       assert_grad(fun, 0, submode="container")(u, v)
       assert_grad(fun, 1, submode="container")(u, v)


   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]], 
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_csum(self, shapes, inds):

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes
           )

       def fun(x, y):
           return tc.csum(lambda a, b: a @ b, x, y)

       assert_grad(fun, 0, submode="container")(w.container, w.container)
       assert_grad(fun, 1, submode="container")(w.container, w.container)


   def test_csum_001(self):

       inds   = ["ijk",   "klm",   "mqlj"]
       shapes = [(3,4,6), (6,2,5), (5,7,2,4)]

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes
           )

       u = ContainerGen(ContainerGen(w.tensors[0], w.tensors[2]), w.tensors[1])
       v = ContainerGen(ContainerGen(w.tensors[0], w.tensors[2]), w.tensors[1])

       def fun(x, y):
           return tc.csum(lambda a, b: a @ b, x, y)

       assert_grad(fun, 0, submode="container")(u, v)
       assert_grad(fun, 1, submode="container")(u, v)













