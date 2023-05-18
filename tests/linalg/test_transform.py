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
import tadpole.linalg   as la

import tadpole.linalg.transform as lat

import tests.linalg.fakes as fake
import tests.linalg.data  as data

from tests.common import (
   available_backends,
)

from tadpole.index import (
   Index,
   IndexGen,  
   Indices,
)




###############################################################################
###                                                                         ###
###  Tensor linalg transforms                                               ###
###                                                                         ###
###############################################################################


# --- Linear algebra transformations ---------------------------------------- #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestLinalgTransforms:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   @pytest.mark.parametrize("shapes, inds, outshape, outinds, which, axis", [
      [[(4,4), (4,4)       ], ["ij", "ij"      ], (8,4),  "lr", None,    0], 
      [[(4,4), (4,4)       ], ["ij", "ij"      ], (8,4),  "lr", "left",  0], 
      [[(4,4), (4,4)       ], ["ij", "ij"      ], (4,8),  "lr", "right", 1],
      [[(4,4), (5,4)       ], ["ij", "kj"      ], (9,4),  "lr", None,    0],  
      [[(4,4), (5,4)       ], ["ij", "kj"      ], (9,4),  "lr", "left",  0], 
      [[(4,4), (4,5)       ], ["ij", "ik"      ], (4,9),  "lr", "right", 1], 
      [[(4,4), (5,4), (6,4)], ["ij", "kj", "lj"], (15,4), "lr", None,    0], 
      [[(4,4), (5,4), (6,4)], ["ij", "kj", "lj"], (15,4), "lr", "left",  0], 
      [[(4,4), (4,5), (4,6)], ["ij", "ik", "il"], (4,15), "lr", "right", 1], 
   ])   
   def test_concat(self, shapes, inds, outshape, outinds, which, axis):

       w = data.ntensor_dat(data.randn)(
              self.backend, inds, shapes
           )
       v = data.indices_dat(outinds, outshape)

       if   which is None:
            out = lat.concat(w.tensors, v.inds)
       else:
            out = lat.concat(w.tensors, v.inds, which=which)

       ans = ar.concat(w.arrays, axis=axis)
       ans = tn.TensorGen(ans, v.inds)

       assert out == ans




