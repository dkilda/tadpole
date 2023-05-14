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
   IndexLit,
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
      data.sparsegrad_dat,
      data.sparsegrad_dat_001,
      data.sparsegrad_dat_002,
      data.sparsegrad_dat_003,
      data.sparsegrad_dat_004,
      data.sparsegrad_dat_005,
   ])
   def test_copy(self, graddat):

       w   = graddat(self.backend)
       out = w.grad.copy()

       assert out == w.grad
       assert out is not w.grad


   @pytest.mark.parametrize("graddat", [
      data.sparsegrad_dat,
      data.sparsegrad_dat_001,
      data.sparsegrad_dat_002,
      data.sparsegrad_dat_003,
      data.sparsegrad_dat_004,
      data.sparsegrad_dat_005,
   ])
   def test_todense(self, graddat):

       w   = graddat(self.backend)
       out = w.grad.todense()

       assert out == w.tensor


   @pytest.mark.parametrize("graddat", [
      data.sparsegrad_dat,
      data.sparsegrad_dat_001,
      data.sparsegrad_dat_002,
      data.sparsegrad_dat_003,
      data.sparsegrad_dat_004,
      data.sparsegrad_dat_005,
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
      data.sparsegrad_dat,
      data.sparsegrad_dat_001,
      data.sparsegrad_dat_002,
      data.sparsegrad_dat_003,
      data.sparsegrad_dat_004,
      data.sparsegrad_dat_005,
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
      data.sparsegrad_dat,
      data.sparsegrad_dat_001,
      data.sparsegrad_dat_002,
      data.sparsegrad_dat_003,
      data.sparsegrad_dat_004,
      data.sparsegrad_dat_005,
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
      data.sparsegrad_dat,
      data.sparsegrad_dat_001,
      data.sparsegrad_dat_002,
      data.sparsegrad_dat_003,
      data.sparsegrad_dat_004,
      data.sparsegrad_dat_005,
   ])
   def test_item(self, graddat):

       w  = graddat(self.backend)

       for pos in itertools.product(*map(range, w.shape)):
           assert w.grad.item(*pos) == w.array.item(*pos)

 
   # --- Tensor properties --- #

   @pytest.mark.parametrize("graddat", [
      data.sparsegrad_dat,
      data.sparsegrad_dat_001,
      data.sparsegrad_dat_002,
      data.sparsegrad_dat_003,
      data.sparsegrad_dat_004,
      data.sparsegrad_dat_005,
   ])
   def test_dtype(self, graddat):

       w = graddat(self.backend)
       assert w.grad.dtype == w.dtype


   @pytest.mark.parametrize("graddat", [
      data.sparsegrad_dat,
      data.sparsegrad_dat_001,
      data.sparsegrad_dat_002,
      data.sparsegrad_dat_003,
      data.sparsegrad_dat_004,
      data.sparsegrad_dat_005,
   ])
   def test_size(self, graddat):

       w = graddat(self.backend)
       assert w.grad.size == np.prod(w.shape)


   @pytest.mark.parametrize("graddat", [
      data.sparsegrad_dat,
      data.sparsegrad_dat_001,
      data.sparsegrad_dat_002,
      data.sparsegrad_dat_003,
      data.sparsegrad_dat_004,
      data.sparsegrad_dat_005,
   ])
   def test_ndim(self, graddat):

       w = graddat(self.backend)
       assert w.grad.ndim == len(w.shape)


   @pytest.mark.parametrize("graddat", [
      data.sparsegrad_dat,
      data.sparsegrad_dat_001,
      data.sparsegrad_dat_002,
      data.sparsegrad_dat_003,
      data.sparsegrad_dat_004,
      data.sparsegrad_dat_005,
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


   # --- Tensor manipulation --- # 

   @pytest.mark.parametrize("shape, indnames, indnames1, inds1", [
      [(2,3,4), "ijk", "aib",  ( 
                                IndexLit("a",2), 
                                IndexLit("i",3), 
                                IndexLit("b",4),
                               )],
   ])
   def test_call(self, shape, indnames, indnames1, inds1):

       w = data.tensor_dat(data.randn)(
              self.backend, indnames, shape
           ) 
       ans = tn.TensorGen(w.array, inds1)

       assert w.tensor(*indnames1) == ans


   @pytest.mark.parametrize("shape, indnames, shape1, indnames1, inds1", [
      [(2,3,4), "ijk", (2,1,3,4), "a1ib", (
                                           IndexLit("a",2), 
                                           IndexLit("1",1), 
                                           IndexLit("i",3), 
                                           IndexLit("b",4),
                                          )],
   ])
   def test_call_001(self, shape, indnames, shape1, indnames1, inds1):

       x = data.tensor_dat(data.randn)(
              self.backend, indnames, shape, seed=1
           ) 
       y = data.array_dat(data.randn)(
              self.backend, shape1, seed=1
           ) 
       ans = tn.TensorGen(y.array, inds1)

       assert x.tensor(*indnames1) == ans


   @pytest.mark.parametrize("shape, inds, inds1", [
      [(2,3,4), "ijk", (
                        IndexLit("a",2), 
                        IndexLit("i",3), 
                        IndexLit("b",4),
                       )],
   ])
   def test_call_002(self, shape, inds, inds1):

       w = data.tensor_dat(data.randn)(
              self.backend, inds, shape
           ) 
       ans = tn.TensorGen(w.array, inds1)

       assert w.tensor(*inds1) == ans


   @pytest.mark.parametrize("indnames, shape", [
      ["ijk", (2,3,4)],
   ])
   def test_C(self, indnames, shape):

       w = data.tensor_dat(data.randn)(
              self.backend, indnames, shape
           )

       out = w.tensor.C
       ans = ar.conj(w.array)
       ans = tn.TensorGen(ans, w.inds)

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("inds, shape, outinds, outaxes", [
      ["ijkl", (2,3,4,5), "lkji", (3,2,1,0)],
   ])
   def test_T(self, inds, shape, outinds, outaxes):

       w = data.tensor_dat(data.randn)(
              self.backend, inds, shape
           ) 

       out = w.tensor.T 
       ans = ar.transpose(w.array, outaxes) 
       ans = tn.TensorGen(ans, w.inds.map(*outinds))

       assert out == ans


   @pytest.mark.parametrize("inds, shape, outinds, outaxes", [
      ["ijkl", (2,3,4,5), "lkji", (3,2,1,0)],
   ])
   def test_H(self, inds, shape, outinds, outaxes):

       w = data.tensor_dat(data.randn)(
              self.backend, inds, shape
           ) 

       out = w.tensor.H 
       ans = ar.transpose(ar.conj(w.array), outaxes) 
       ans = tn.TensorGen(ans, w.inds.map(*outinds))

       assert out == ans        




