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

import tadpole.array.backends        as backends
import tadpole.tensor.elemwise_unary as tnu
import tadpole.tensor.engine         as tne 

import tests.tensor.fakes as fake
import tests.tensor.data  as data


from tests.common import (
   available_backends,
)


from tadpole.tensor.types import (
   Pluggable,
   Tensor, 
   Space,
)


from tadpole.index import (
   Index,
   IndexGen,  
   Indices,
)




###############################################################################
###                                                                         ###
###  Special tensor types for gradients                                     ###
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

       
   # --- Basic functionality --- #

   @pytest.mark.parametrize("graddat", [
      data.nullgrad_dat_001,
   ])
   def test_copy(self, graddat):

       w   = graddat(self.backend)
       out = w.grad.copy()

       assert out == w.grad
       assert out is not w.grad


   @pytest.mark.parametrize("graddat", [
      data.nullgrad_dat_001,
   ])
   def test_todense(self, graddat):

       w   = graddat(self.backend)
       out = w.grad.todense()

       assert out == w.tensor


   @pytest.mark.parametrize("graddat", [
      data.nullgrad_dat_001,
   ])
   def test_tonull(self, graddat):

       w   = graddat(self.backend)
       out = w.grad.tonull()

       ans = tn.TensorSpace(
                ar.ArraySpace(w.backend, w.shape, w.dtype), 
                w.inds
             )
       ans = tn.NullGrad(ans) 

       assert out == ans


   @pytest.mark.parametrize("graddat", [
      data.nullgrad_dat_001,
   ])
   def test_withdata(self, graddat):

       w = graddat(self.backend)
       x = data.array_dat(data.randn)(
              self.backend, w.shape, seed=1
           )
   
       out = w.grad.withdata(x.array)
       ans = tn.TensorGen(x.array, w.inds)

       assert out == ans


   @pytest.mark.parametrize("graddat", [
      data.nullgrad_dat_001,
   ])
   def test_space(self, graddat):

       w   = graddat(self.backend)
       out = w.grad.space()
       ans = tn.TensorSpace(
                ar.ArraySpace(w.backend, w.shape, w.dtype), 
                w.inds
             )

       assert out == ans


   @pytest.mark.parametrize("graddat", [
      data.nullgrad_dat_001,
   ])
   def test_item(self, graddat):

       w = graddat(self.backend)

       for pos in itertools.product(*map(range, w.shape)):
           assert w.grad.item(*pos) == w.array.item(*pos)

 
   # --- Tensor properties --- #

   @pytest.mark.parametrize("graddat", [
      data.nullgrad_dat_001,
   ])
   def test_dtype(self, graddat):

       w = graddat(self.backend)
       assert w.grad.dtype == w.dtype


   @pytest.mark.parametrize("graddat", [
      data.nullgrad_dat_001,
   ])
   def test_size(self, graddat):

       w = graddat(self.backend)
       assert w.grad.size == np.prod(w.shape)


   @pytest.mark.parametrize("graddat", [
      data.nullgrad_dat_001,
   ])
   def test_ndim(self, graddat):

       w = graddat(self.backend)
       assert w.grad.ndim == len(w.shape)


   @pytest.mark.parametrize("graddat", [
      data.nullgrad_dat_001,
   ])
   def test_shape(self, graddat):

       w = graddat(self.backend)
       assert w.grad.shape == w.shape




# --- Sparse gradient ------------------------------------------------------- #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestSparseGrad:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend

       
   # --- Basic functionality --- #

   @pytest.mark.parametrize("graddat", [
      data.sparsegrad_dat_001,
   ])
   def test_copy(self, graddat):

       w   = graddat(self.backend)
       out = w.grad.copy()

       assert out == w.grad
       assert out is not w.grad


   @pytest.mark.parametrize("graddat", [
      data.sparsegrad_dat_001,
   ])
   def test_todense(self, graddat):

       w   = graddat(self.backend)
       out = w.grad.todense()

       assert out == w.tensor


   @pytest.mark.parametrize("graddat", [
      data.sparsegrad_dat_001,
   ])
   def test_tonull(self, graddat):

       w   = graddat(self.backend)
       out = w.grad.tonull()

       ans = tn.TensorSpace(
                ar.ArraySpace(w.backend, w.shape, w.dtype), 
                w.inds
             )
       ans = tn.NullGrad(ans)

       assert out == ans


   @pytest.mark.parametrize("graddat", [
      data.sparsegrad_dat_001,
   ])
   def test_withdata(self, graddat):

       w = graddat(
              self.backend, seed=1
           )
       x = data.array_dat(data.randn)(
              self.backend, w.shape, seed=2
           )
   
       out = w.grad.withdata(x.array)
       ans = tn.TensorGen(x.array, w.inds)

       assert out == ans


   @pytest.mark.parametrize("graddat", [
      data.sparsegrad_dat_001,
   ])
   def test_space(self, graddat):

       w   = graddat(self.backend)
       out = w.grad.space()
       ans = tn.TensorSpace(
                ar.ArraySpace(w.backend, w.shape, w.dtype), 
                w.inds
             )

       assert out == ans


   @pytest.mark.parametrize("graddat", [
      data.sparsegrad_dat_001,
   ])
   def test_item(self, graddat):

       w  = graddat(self.backend)

       for pos in itertools.product(*map(range, w.shape)):
           assert w.grad.item(*pos) == w.array.item(*pos)

 
   # --- Tensor properties --- #

   @pytest.mark.parametrize("graddat", [
      data.sparsegrad_dat_001,
   ])
   def test_dtype(self, graddat):

       w = graddat(self.backend)
       assert w.grad.dtype == w.dtype


   @pytest.mark.parametrize("graddat", [
      data.sparsegrad_dat_001,
   ])
   def test_size(self, graddat):

       w = graddat(self.backend)
       assert w.grad.size == np.prod(w.shape)


   @pytest.mark.parametrize("graddat", [
      data.sparsegrad_dat_001,
   ])
   def test_ndim(self, graddat):

       w = graddat(self.backend)
       assert w.grad.ndim == len(w.shape)


   @pytest.mark.parametrize("graddat", [
      data.sparsegrad_dat_001,
   ])
   def test_shape(self, graddat):

       w = graddat(self.backend)
       assert w.grad.shape == w.shape




###############################################################################
###                                                                         ###
###  General tensor                                                         ###
###                                                                         ###
###############################################################################


# --- General tensor -------------------------------------------------------- #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestTensorGen:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   # --- Tensor factories --- #

   @pytest.mark.parametrize("inds, shape, dtype", [
      ["ijk", (2,3,4), "complex128"],
   ])
   def test_astensor(self, inds, shape, dtype):

       w = data.tensor_dat(data.randn)(
              self.backend, inds, shape, dtype=dtype
           )       

       assert tn.astensor(w.tensor) is w.tensor
       assert tn.astensor(w.array, w.inds) == w.tensor
       assert tn.astensor(w.data,  w.inds) == w.tensor


   # --- Basic functionality --- #

   @pytest.mark.parametrize("inds, shape, dtype", [
      ["ijk", (2,3,4), "complex128"],
   ])
   def test_copy(self, inds, shape, dtype):

       w = data.tensor_dat(data.randn)(
              self.backend, inds, shape, dtype=dtype
           )
       out = w.tensor.copy()

       assert out == w.tensor
       assert out is not w.tensor


   @pytest.mark.parametrize("inds, shape, dtype", [
      ["ijk", (2,3,4), "complex128"],
   ])
   def test_todense(self, inds, shape, dtype):

       w = data.tensor_dat(data.randn)(
              self.backend, inds, shape, dtype=dtype
           )
       out = w.tensor.todense()

       assert out == w.tensor


   @pytest.mark.parametrize("inds, shape, dtype", [
      ["ijk", (2,3,4), "complex128"],
   ])
   def test_tonull(self, inds, shape, dtype):

       w = data.tensor_dat(data.randn)(
              self.backend, inds, shape, dtype=dtype
           )
       out = w.tensor.tonull()

       ans = tn.TensorSpace(
                ar.ArraySpace(w.backend, w.shape, w.dtype), 
                w.inds
             )
       ans = tn.NullGrad(ans)

       assert out == ans


   @pytest.mark.parametrize("inds, shape, dtype", [
      ["ijk", (2,3,4), "complex128"],
   ])
   def test_withdata(self, inds, shape, dtype):

       w = data.tensor_dat(data.randn)(
              self.backend, inds, shape, dtype=dtype, seed=1
           )
       x = data.array_dat(data.randn)(
              self.backend, w.shape, dtype=w.dtype, seed=2
           )
   
       out = w.tensor.withdata(x.array)
       ans = tn.TensorGen(x.array, w.inds)

       assert out == ans


   @pytest.mark.parametrize("inds, shape, dtype", [
      ["ijk", (2,3,4), "complex128"],
   ])
   def test_space(self, inds, shape, dtype):

       w = data.tensor_dat(data.randn)(
              self.backend, inds, shape, dtype=dtype
           )
       out = w.tensor.space()
       ans = tn.TensorSpace(
                ar.ArraySpace(w.backend, w.shape, w.dtype), 
                w.inds
             )

       assert out == ans


   @pytest.mark.parametrize("inds, shape, dtype", [
      ["ijk", (2,3,4), "complex128"],
   ])
   def test_item(self, inds, shape, dtype):

       w = data.tensor_dat(data.randn)(
              self.backend, inds, shape, dtype=dtype
           )

       for pos in itertools.product(*map(range, w.shape)):
           assert w.tensor.item(*pos) == w.array.item(*pos)

 
   # --- Tensor properties --- #

   @pytest.mark.parametrize("inds, shape, dtype", [
      ["ijk", (2,3,4), "complex128"],
   ])
   def test_dtype(self, inds, shape, dtype):

       w = data.tensor_dat(data.randn)(
              self.backend, inds, shape, dtype=dtype
           )
       assert w.tensor.dtype == w.dtype


   @pytest.mark.parametrize("inds, shape, dtype", [
      ["ijk", (2,3,4), "complex128"],
   ])
   def test_size(self, inds, shape, dtype):

       w = data.tensor_dat(data.randn)(
              self.backend, inds, shape, dtype=dtype
           )
       assert w.tensor.size == np.prod(w.shape)


   @pytest.mark.parametrize("inds, shape, dtype", [
      ["ijk", (2,3,4), "complex128"],
   ])
   def test_ndim(self, inds, shape, dtype):

       w = data.tensor_dat(data.randn)(
              self.backend, inds, shape, dtype=dtype
           )
       assert w.tensor.ndim == len(w.shape)


   @pytest.mark.parametrize("inds, shape, dtype", [
      ["ijk", (2,3,4), "complex128"],
   ])
   def test_shape(self, inds, shape, dtype):

       w = data.tensor_dat(data.randn)(
              self.backend, inds, shape, dtype=dtype
           )
       assert w.tensor.shape == w.shape





















