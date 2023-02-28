#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import itertools
import numpy as np

import tests.array.fakes as fake
import tests.array.data  as data

import tadpole.util           as util
import tadpole.array.backends as backends
import tadpole.array.core     as core
import tadpole.array.function as function




###############################################################################
###                                                                         ###
###  A general framework for array gradients.                               ###
###                                                                         ###
###############################################################################


# --- Sparse gradient class ------------------------------------------------- #

class TestSparseGrad:

   @pytest.mark.parametrize("backend", ["numpy"])
   @pytest.mark.parametrize("dtype, shape, idxs, vals", [
      [
       "float64", 
       (2,3,4), 
       ((1,0,2), (0,2,1), (1,0,3)), 
       np.random.randn(3),
      ],
      [
       "complex128", 
       (2,3,4), 
       ((1,0,2), (0,2,1), (1,0,3)), 
       np.random.randn(3) + 1j * np.random.randn(3),
      ],
   ])
   def test_todense(self, backend, dtype, shape, idxs, vals):


       idxs = list(idxs)
       vals = np.random.randn(3)
       vals = [vals[i] for i in range(3)]

       x = data.sparse_grad_dat(backend, shape, dtype, idxs, vals)
       assert x.grad.todense() == x.dense


































