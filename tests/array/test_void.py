#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import itertools

import numpy as np
import scipy

import tests.array.fakes as fake
import tests.array.data  as data

import tadpole.util           as util
import tadpole.array          as ar
import tadpole.array.space    as sp
import tadpole.array.unary    as unary
import tadpole.array.binary   as binary
import tadpole.array.nary     as nary
import tadpole.array.void     as void
import tadpole.array.backends as backends


from tests.common import (
   options,
   available_backends,
)




###############################################################################
###                                                                         ###
###  Definition of Void Array (supports array creation)                     ###
###                                                                         ###
###############################################################################


# --- Void Array ------------------------------------------------------------ #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestArray:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   # --- Array methods --- #

   @pytest.mark.parametrize("shape", [(2,3,4)])
   @pytest.mark.parametrize("dtype", ["complex128"])
   def test_new(self, shape, dtype):

       w = data.narray_dat(data.randn)(self.backend)
       x = data.array_dat(data.randn)(
              self.backend, shape, dtype=dtype
           )

       assert w.narray.new(x.data) == x.array


   def test_nary(self):

       w = data.narray_dat(data.randn)(self.backend)

       assert w.narray.nary() == nary.Array(w.backend)


   @pytest.mark.parametrize("shapes", [
      [(2,3,4), (5,4,6), (5,3,2)],
   ])
   def test_or(self, shapes):

       x = data.narray_dat(data.randn)(self.backend)
       y = data.narray_dat(data.randn)(self.backend, shapes)

       assert x.narray | y.narray is y.narray


   # --- Data type methods --- #

   @pytest.mark.parametrize("dtype, iscomplex", [
      ["complex128", True],
      ["float64",    False],
   ])
   def test_iscomplex_type(self, dtype, iscomplex):

       assert ar.iscomplex_type(dtype) == iscomplex

       
   # --- Array creation methods --- #

   @pytest.mark.parametrize("sampledat", [
      data.zeros_dat_001,
   ])
   def test_zeros(self, sampledat):

       x   = sampledat(self.backend)
       out = ar.zeros(
                x.shape, **options(dtype=x.dtype, backend=self.backend)
             )

       assert out == x.array


   @pytest.mark.parametrize("sampledat", [
      data.ones_dat_001,
   ])
   def test_ones(self, sampledat):

       x   = sampledat(self.backend)
       out = ar.ones(
                x.shape, **options(dtype=x.dtype, backend=self.backend)
             )

       assert out == x.array


   @pytest.mark.parametrize("basisdat", [
      data.basis_real_dat_001,
   ])
   def test_unit(self, basisdat):

       x = basisdat(self.backend)

       for idx, array in zip(x.idxs, x.arrays):

           out = ar.unit(
                    x.shape, 
                    idx, 
                    **options(dtype=x.dtype, backend=self.backend)
                 )

           assert out == array


   @pytest.mark.parametrize("sampledat", [
      data.rand_real_dat_001,
      data.rand_complex_dat_001,
   ])
   def test_rand(self, sampledat):

       seed = 1
       x    = sampledat(self.backend, seed=seed)
       out  = ar.rand(
                 x.shape, 
                 **options(dtype=x.dtype, seed=seed, backend=self.backend)
              )

       assert out == x.array


   @pytest.mark.parametrize("sampledat", [
      data.randn_real_dat_001,
      data.randn_complex_dat_001,
   ])
   def test_randn(self, sampledat):

       seed = 1
       x    = sampledat(self.backend, seed=seed)
       out  = ar.randn(
                 x.shape, 
                 **options(dtype=x.dtype, seed=seed, backend=self.backend)
              )

       assert out == x.array


   @pytest.mark.parametrize("sampledat", [
      data.randuniform_real_dat_001,
      data.randuniform_complex_dat_001,
   ])
   def test_randuniform(self, sampledat):

       seed = 1
       x    = sampledat(self.backend, seed=seed)
       out  = ar.randuniform(
                 x.shape, 
                 x.opts["boundaries"],
                 **options(dtype=x.dtype, seed=seed, backend=self.backend)
              )

       assert out == x.array


   @pytest.mark.parametrize("N, M, k", [
      [4, None, None],
      [3, 4,    None],
      [5, 4,       1],
      [5, 4,      -1],
   ])
   def test_eye(self, N, M, k):

       out = ar.eye(N, **options(M=M, backend=self.backend, k=k))
       ans = np.eye(N, **options(M=M, k=k))
       ans = unary.asarray(ans, **options(backend=self.backend))
 
       assert out == ans











