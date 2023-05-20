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
   def test_ascontainer(self, shapes, inds):

       def fun(*xs):
           return tc.ascontainer(*xs)

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
   def test_sparsegrad(self, shapes, inds):

       positions = {
          1: [0,       slice(0,1),                       ],
          2: [0, 1,    slice(0,1), slice(0,2), slice(1,2)],
          3: [0, 1, 2, slice(0,1), slice(0,2), slice(1,2), 
                       slice(1,3), slice(2,3), slice(0,3)],
       }[len(shapes)]

       def fun(x, pos, space):
           return tc.sparsegrad(x, pos, space)

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes
           )

       for pos in positions:

           x = w.tensors[pos]

           if isinstance(pos, slice):
              x = ContainerGen(x)

           assert_grad(fun, submode="container")(x, pos, w.space)





