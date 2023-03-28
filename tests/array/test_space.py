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
)


# TODO use this: 
# pytest util autodiff global array

###############################################################################
###                                                                         ###
###  Definition of Array Space                                              ###
###                                                                         ###
###############################################################################


# --- Array Space ----------------------------------------------------------- #

@pytest.mark.parametrize("current_backend", ["numpy_backend"], indirect=True)
class TestArraySpace:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   # --- Create Array Space --- #

   @pytest.mark.parametrize("shape", [(2,3,4)])
   @pytest.mark.parametrize("dtype", ["complex128"])   
   def test_arrayspace(self, shape, dtype):

       w   = data.arrayspace_dat(self.backend, shape, dtype)
       out = ar.arrayspace(w.shape, w.dtype, backend=w.backend.name())

       assert out == w.space


   # --- Space properties --- #

   @pytest.mark.parametrize("shape", [(2,3,4)])
   @pytest.mark.parametrize("dtype", ["complex128"])   
   def test_dtype(self, shape, dtype):

       w   = data.arrayspace_dat(self.backend, shape, dtype)
       out = ar.arrayspace(w.shape, w.dtype, backend=w.backend.name())

       assert w.space.dtype == w.dtype 


   @pytest.mark.parametrize("shape", [(2,3,4)])
   @pytest.mark.parametrize("dtype", ["complex128"])   
   def test_size(self, shape, dtype):

       w   = data.arrayspace_dat(self.backend, shape, dtype)
       out = ar.arrayspace(w.shape, w.dtype, backend=w.backend.name())

       assert w.space.size == np.prod(w.shape)


   @pytest.mark.parametrize("shape", [(2,3,4)])
   @pytest.mark.parametrize("dtype", ["complex128"])   
   def test_ndim(self, shape, dtype):

       w   = data.arrayspace_dat(self.backend, shape, dtype)
       out = ar.arrayspace(w.shape, w.dtype, backend=w.backend.name())

       assert w.space.ndim == len(w.shape)


   @pytest.mark.parametrize("shape", [(2,3,4)])
   @pytest.mark.parametrize("dtype", ["complex128"])   
   def test_shape(self, shape, dtype):

       w   = data.arrayspace_dat(self.backend, shape, dtype)
       out = ar.arrayspace(w.shape, w.dtype, backend=w.backend.name())

       assert w.space.shape == w.shape


   # --- Array generators --- #

   @pytest.mark.parametrize("basisdat", [
      data.basis_real_dat_001,
   ])
   def test_units(self, basisdat):

       w = basisdat(self.backend)
       x = data.arrayspace_dat(self.backend, w.shape, w.dtype)
 
       out = x.space.units()

       assert list(out) == list(w.arrays)


   @pytest.mark.parametrize("basisdat", [
      data.basis_real_dat_001, 
      data.basis_complex_dat_001,
   ])
   def test_basis(self, basisdat):

       w = basisdat(self.backend)
       x = data.arrayspace_dat(self.backend, w.shape, w.dtype)
 
       out = x.space.basis()

       assert list(out) == list(w.arrays)


















































