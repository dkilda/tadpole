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

   @pytest.mark.parametrize("backend",  ["numpy"])
   @pytest.mark.parametrize("graddats", [
      (data.sparse_grad_dat_001,                         ),
      (data.sparse_grad_dat_002,                         ),
      (data.sparse_grad_dat_001, data.sparse_grad_dat_002),
   ])
   def test_pluginto(self, backend, graddats):

       ws  = [dat(backend)           for dat in graddats] 
       seq = [(w.dense, w.densedata) for w   in ws]

       fun     = fake.Fun(None)
       ans     = function.FunCall(fun, util.Sequence(seq))
       funcall = function.FunCall(fun)

       for w in ws:
           funcall = w.grad.pluginto(funcall)

       assert funcall == ans


   @pytest.mark.parametrize("backend",  ["numpy"])
   @pytest.mark.parametrize("graddat", [
      data.sparse_grad_dat_001,
      data.sparse_grad_dat_002,
   ])
   def test_add_zero(self, backend, graddat):

       w = graddat(backend)
       assert w.grad + 0 == w.dense


   @pytest.mark.parametrize("backend",  ["numpy"])
   @pytest.mark.parametrize("graddat", [
      data.sparse_grad_dat_001,
      data.sparse_grad_dat_002,
   ])
   def test_add_dense(self, backend, graddat):

       w = graddat(backend)
       x = data.array_dat(data.randn)(backend, w.shape, dtype=w.dtype)

       assert w.grad + x.array == w.dense + x.array


   @pytest.mark.parametrize("backend",            ["numpy"])
   @pytest.mark.parametrize("graddat1, graddat2", [
      (data.sparse_grad_dat_001, data.sparse_grad_dat_001),
   ])
   def test_add_sparse(self, backend, graddat1, graddat2):

       x = graddat1(backend)
       y = graddat2(backend)

       assert x.grad + 2 * y.grad == x.dense + 2 * y.dense


   @pytest.mark.parametrize("backend", ["numpy"])
   @pytest.mark.parametrize("graddat", [
      data.sparse_grad_dat_001,
      data.sparse_grad_dat_002,
   ])
   def test_todense(self, backend, graddat):

       x = graddat(backend)
       assert x.grad.todense() == x.dense


   @pytest.mark.parametrize("backend", ["numpy"])
   @pytest.mark.parametrize("graddat", [
      data.sparse_grad_dat_001,
      data.sparse_grad_dat_002,
   ])
   def test_copy(self, backend, graddat):

       x = graddat(backend)
       assert x.grad.copy() == x.dense


   @pytest.mark.parametrize("backend", ["numpy"])
   @pytest.mark.parametrize("graddat", [
      data.sparse_grad_dat_001,
      data.sparse_grad_dat_002,
   ])
   def test_space(self, backend, graddat):

       x = graddat(backend)
       assert x.grad.space() == x.space


   @pytest.mark.parametrize("backend", ["numpy"])
   @pytest.mark.parametrize("graddat", [
      data.sparse_grad_dat_001,
      data.sparse_grad_dat_002,
   ])
   def test_dtype(self, backend, graddat):

       x = graddat(backend)
       assert x.grad.dtype == x.dtype


   @pytest.mark.parametrize("backend", ["numpy"])
   @pytest.mark.parametrize("graddat", [
      data.sparse_grad_dat_001,
      data.sparse_grad_dat_002,
   ])
   def test_ndim(self, backend, graddat):

       x = graddat(backend)
       assert x.grad.ndim == len(x.shape)


   @pytest.mark.parametrize("backend", ["numpy"])
   @pytest.mark.parametrize("graddat", [
      data.sparse_grad_dat_001,
      data.sparse_grad_dat_002,
   ])
   def test_shape(self, backend, graddat):

       x = graddat(backend)
       assert x.grad.shape == x.shape


   @pytest.mark.parametrize("backend", ["numpy"])
   @pytest.mark.parametrize("graddat", [
      data.sparse_grad_dat_001,
      data.sparse_grad_dat_002,
   ])
   def test_getitem(self, backend, graddat):

       x = graddat(backend)

       for idx in itertools.product(*map(range, x.shape)):
           assert x.grad[idx] == x.dense[idx]




