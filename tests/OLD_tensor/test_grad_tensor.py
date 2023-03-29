#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import itertools
import numpy as np

import tests.tensor.fakes as fake
import tests.tensor.data  as data

import tadpole.util              as util
import tadpole.tensor.core       as core
import tadpole.tensor.grad       as grad
import tadpole.tensor.backends   as backends
import tadpole.tensor.function   as function
import tadpole.tensor.operations as op

import tadpole.tensor as tn




###############################################################################
###                                                                         ###
###  A general framework for tensor gradients.                              ###
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

       ws  = [dat(backend)             for dat in graddats] 
       seq = [(w.backend, w.densedata) for w   in ws]

       fun     = fake.Fun(None)
       ans     = function.TransformCall(fun, util.Sequence(seq))
       funcall = function.TransformCall(fun)

       for w in ws:
           funcall = w.grad.pluginto(funcall)

       assert funcall == ans


   @pytest.mark.parametrize("backend", ["numpy"])
   @pytest.mark.parametrize("graddat", [
      data.sparse_grad_dat_001,
      data.sparse_grad_dat_002,
   ])
   def test_addto_zero(self, backend, graddat):

       w = graddat(backend)

       out = w.grad.addto(grad.ZeroGrad())
       assert tn.allclose(out, w.dense)


   @pytest.mark.parametrize("backend", ["numpy"])
   @pytest.mark.parametrize("graddat", [
      data.sparse_grad_dat_001,
      data.sparse_grad_dat_002,
   ])
   def test_addto_dense(self, backend, graddat):

       w = graddat(backend)
       x = data.tensor_dat(data.randn)(backend, w.shape, dtype=w.dtype)

       out = w.grad.addto(x.tensor)
       assert tn.allclose(out, w.dense + x.tensor)


   @pytest.mark.parametrize("backend",            ["numpy"])
   @pytest.mark.parametrize("graddat1, graddat2", [
      (data.sparse_grad_dat_001, data.sparse_grad_dat_001),
   ])
   def test_addto_sparse(self, backend, graddat1, graddat2):

       x = graddat1(backend)
       y = graddat2(backend)

       out = x.grad.addto(y.grad)
       assert tn.allclose(out, x.dense + y.dense)


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
       assert x.grad.copy() == x.grad


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
   def test_size(self, backend, graddat):

       x = graddat(backend)
       assert x.grad.size == len(x.vals)


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



