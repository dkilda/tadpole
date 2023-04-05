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

import tadpole.array.backends         as backends
import tadpole.tensor.elemwise_binary as tnb
import tadpole.tensor.engine          as tne 

import tests.tensor.fakes as fake
import tests.tensor.data  as data


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
###  Tensor binary elementwise engine and operator                          ###
###                                                                         ###
###############################################################################


# --- Helper: unary typecast binary ----------------------------------------- #

@pytest.mark.parametrize("current_backend", ["numpy_backend"], indirect=True)
class TestTypecastBinary:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   @pytest.mark.parametrize("fundat",  [
       data.binary_wrappedfun_dat_001,
       data.binary_wrappedfun_dat_002,
       data.binary_wrappedfun_dat_003,
       data.binary_wrappedfun_dat_004,
   ])
   def test_typecast_binary(self, fundat):

       w = fundat(self.backend)
       assert tn.allclose(w.wrappedfun(*w.args), w.out)




# --- Tensor binary elementwise operator ------------------------------------ #  

@pytest.mark.parametrize("current_backend", ["numpy_backend"], indirect=True)
class TestTensorElemwiseBinary:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   # --- Gradient accumulation --- #

   @pytest.mark.parametrize("indnames, shape", [
      ["ijk", (2,3,4)],
   ])
   def test_addgrads(self, indnames, shape):

       x = data.tensor_dat(data.randn)(
              self.backend, indnames, shape, seed=1
           )
       y = data.tensor_dat(data.randn)(
              self.backend, indnames, shape, seed=2
           )

       xtensor = x.tensor
       ytensor = tn.TensorGen(y.array, x.inds)

       out = tn.addgrads(xtensor, ytensor)
       ans = ar.add(x.array, y.array)
       ans = tn.TensorGen(ans, x.inds)

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("graddat", [
      data.sparsegrad_dat_001,
   ])
   @pytest.mark.parametrize("dtype", [
      "complex128",
      "float64"
   ])
   def test_addgrads_sparse_dense(self, graddat, dtype):

       w = graddat(
              self.backend, dtype, seed=1
           )      
       x = data.array_dat(data.randn)(
              self.backend, w.shape, dtype=dtype, seed=2
           )
   
       xtensor = tn.TensorGen(x.array, w.inds)

       out = tn.addgrads(w.grad, xtensor)
       ans = ar.add(w.array, x.array)
       ans = tn.TensorGen(ans, w.inds)

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("graddat", [
      data.sparsegrad_dat_001,
   ])
   @pytest.mark.parametrize("dtype", [
      "complex128",
      "float64"
   ])
   def test_addgrads_dense_sparse(self, graddat, dtype):

       w = graddat(
              self.backend, dtype, seed=1
           )      
       x = data.array_dat(data.randn)(
              self.backend, w.shape, dtype=dtype, seed=2
           )
   
       xtensor = tn.TensorGen(x.array, w.inds)

       out = tn.addgrads(xtensor, w.grad)
       ans = ar.add(x.array, w.array)
       ans = tn.TensorGen(ans, w.inds)

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("graddat", [
      data.sparsegrad_dat_001,
   ])
   @pytest.mark.parametrize("dtype", [
      "complex128",
      "float64"
   ])
   def test_addgrads_sparse_sparse(self, graddat, dtype):

       x = graddat(self.backend, dtype, seed=1)      
       y = graddat(self.backend, dtype, seed=2) 

       out = tn.addgrads(x.grad, x.grad)
       ans = ar.add(x.array, x.array)
       ans = tn.TensorGen(ans, x.inds)

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("indnames, shape", [
      ["ijk", (2,3,4)],
   ])
   def test_addgrads_dense_null(self, indnames, shape):

       x = data.tensor_dat(data.randn)(
              self.backend, indnames, shape, seed=1
           )

       xtensor = x.tensor
       ytensor = tn.NullGrad(xtensor.space())

       out = tn.addgrads(xtensor, ytensor)
       ans = tn.TensorGen(x.array, x.inds)

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("indnames, shape", [
      ["ijk", (2,3,4)],
   ])
   def test_addgrads_null_dense(self, indnames, shape):

       x = data.tensor_dat(data.randn)(
              self.backend, indnames, shape, seed=1
           )

       xtensor = x.tensor
       ytensor = tn.NullGrad(xtensor.space())

       out = tn.addgrads(ytensor, xtensor)
       ans = tn.TensorGen(x.array, x.inds)

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("indnames, shape", [
      ["ijk", (2,3,4)],
   ])
   def test_addgrads_null_null(self, indnames, shape):

       x = data.tensor_dat(data.randn)(
              self.backend, indnames, shape, seed=1
           )

       xtensor = tn.NullGrad(x.tensor.space())
       ytensor = tn.NullGrad(x.tensor.space())

       out = tn.addgrads(xtensor, ytensor)
       assert out == xtensor


   @pytest.mark.parametrize("graddat", [
      data.sparsegrad_dat_001,
   ])
   @pytest.mark.parametrize("dtype", [
      "complex128",
      "float64"
   ])
   def test_addgrads_sparse_null(self, graddat, dtype):

       w = graddat(self.backend, dtype, seed=1)     

       x   = tn.NullGrad(w.grad.space())
       out = tn.addgrads(w.grad, x)

       assert tn.allclose(out, w.tensor)


   @pytest.mark.parametrize("graddat", [
      data.sparsegrad_dat_001,
   ])
   @pytest.mark.parametrize("dtype", [
      "complex128",
      "float64"
   ])
   def test_addgrads_null_sparse(self, graddat, dtype):

       w = graddat(self.backend, dtype, seed=1)     

       x   = tn.NullGrad(w.grad.space())
       out = tn.addgrads(x, w.grad)

       assert tn.allclose(out, w.tensor)
       

   # --- Standard math --- #

   @pytest.mark.parametrize("indnames, shape", [
      ["ijk", (2,3,4)],
   ])
   def test_add(self, indnames, shape):

       x = data.tensor_dat(data.randn)(
              self.backend, indnames, shape, seed=1
           )
       y = data.tensor_dat(data.randn)(
              self.backend, indnames, shape, seed=2
           )

       xtensor = x.tensor
       ytensor = tn.TensorGen(y.array, x.inds)

       out = xtensor + ytensor
       ans = ar.add(x.array, y.array)
       ans = tn.TensorGen(ans, x.inds)

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("indnames, shape", [
      ["ijk", (2,3,4)],
   ])
   def test_sub(self, indnames, shape):

       x = data.tensor_dat(data.randn)(
              self.backend, indnames, shape, seed=1
           )
       y = data.tensor_dat(data.randn)(
              self.backend, indnames, shape, seed=2
           )

       xtensor = x.tensor
       ytensor = tn.TensorGen(y.array, x.inds)

       out = xtensor - ytensor
       ans = ar.sub(x.array, y.array)
       ans = tn.TensorGen(ans, x.inds)

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("indnames, shape", [
      ["ijk", (2,3,4)],
   ])
   def test_mul(self, indnames, shape):

       x = data.tensor_dat(data.randn)(
              self.backend, indnames, shape, seed=1
           )
       y = data.tensor_dat(data.randn)(
              self.backend, indnames, shape, seed=2
           )

       xtensor = x.tensor
       ytensor = tn.TensorGen(y.array, x.inds)

       out = xtensor * ytensor
       ans = ar.mul(x.array, y.array)
       ans = tn.TensorGen(ans, x.inds)

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("indnames, shape", [
      ["ijk", (2,3,4)],
   ])
   def test_div(self, indnames, shape):

       x = data.tensor_dat(data.randn)(
              self.backend, indnames, shape, seed=1
           )
       y = data.tensor_dat(data.randn)(
              self.backend, indnames, shape, seed=2
           )

       xtensor = x.tensor
       ytensor = tn.TensorGen(y.array, x.inds)

       out = xtensor / ytensor
       ans = ar.div(x.array, y.array)
       ans = tn.TensorGen(ans, x.inds)

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("indnames, shape", [
      ["ijk", (2,3,4)],
   ])
   def test_power(self, indnames, shape):

       x = data.tensor_dat(data.randn)(
              self.backend, indnames, shape, seed=1
           )
       y = data.tensor_dat(data.randn)(
              self.backend, indnames, shape, seed=2
           )

       xtensor = x.tensor
       ytensor = tn.TensorGen(y.array, x.inds)

       out = xtensor ** ytensor
       ans = ar.power(x.array, y.array)
       ans = tn.TensorGen(ans, x.inds)

       assert tn.allclose(out, ans)


   # --- Logical operations --- #

   @pytest.mark.parametrize("indnames, shape, nvals", [
      ["ijk", (2,3,4), 5],
      ["ijk", (2,3,4), 0],
   ])
   def test_allclose(self, indnames, shape, nvals):

       x = data.tensor_dat(data.randn_pos)(
              self.backend, indnames, shape, 
              seed=1, nvals=nvals, defaultval=-2.37+0.58j
           )
       y = data.tensor_dat(data.randn_pos)(
              self.backend, indnames, shape, 
              seed=2, nvals=nvals, defaultval=-2.37+0.58j
           )

       xtensor = x.tensor
       ytensor = tn.TensorGen(y.array, x.inds)

       if   nvals > 0:
            assert not tn.allclose(xtensor, ytensor) 
       else:
            assert tn.allclose(xtensor, ytensor)


   @pytest.mark.parametrize("indnames, shape, nvals", [
      ["ijk", (2,3,4), 5],
      ["ijk", (2,3,4), 0],
   ])
   def test_allequal(self, indnames, shape, nvals):

       x = data.tensor_dat(data.randn_pos)(
              self.backend, indnames, shape, 
              seed=1, nvals=nvals, defaultval=1+0j
           )
       y = data.tensor_dat(data.randn_pos)(
              self.backend, indnames, shape, 
              seed=2, nvals=nvals, defaultval=1+0j
           )

       xtensor = x.tensor
       ytensor = tn.TensorGen(y.array, x.inds)

       if   nvals > 0:
            assert not tn.allequal(xtensor, ytensor) 
       else:
            assert tn.allequal(xtensor, ytensor)


   @pytest.mark.parametrize("indnames, shape, nvals", [
      ["ijk", (2,3,4), 5],
   ])
   def test_isclose(self, indnames, shape, nvals):

       x = data.tensor_dat(data.randn_pos)(
              self.backend, indnames, shape, 
              seed=1, nvals=nvals, defaultval=-2.37+0.58j
           )
       y = data.tensor_dat(data.randn_pos)(
              self.backend, indnames, shape, 
              seed=2, nvals=nvals, defaultval=-2.37+0.58j
           )

       xtensor = x.tensor
       ytensor = tn.TensorGen(y.array, x.inds)

       out = tn.isclose(xtensor, ytensor) 
       ans = ar.isclose(x.array, y.array)
       ans = tn.TensorGen(ans, x.inds)

       assert out == ans


   @pytest.mark.parametrize("indnames, shape, nvals", [
      ["ijk", (2,3,4), 5],
   ])
   def test_isequal(self, indnames, shape, nvals):

       x = data.tensor_dat(data.randn_pos)(
              self.backend, indnames, shape, 
              seed=1, nvals=nvals, defaultval=1+0j
           )
       y = data.tensor_dat(data.randn_pos)(
              self.backend, indnames, shape, 
              seed=2, nvals=nvals, defaultval=1+0j
           )

       xtensor = x.tensor
       ytensor = tn.TensorGen(y.array, x.inds)

       out = tn.isequal(xtensor, ytensor) 
       ans = ar.isequal(x.array, y.array)
       ans = tn.TensorGen(ans, x.inds)

       assert out == ans


   @pytest.mark.parametrize("indnames, shape, nvals", [
      ["ijk", (2,3,4), 5],
   ])
   def test_notequal(self, indnames, shape, nvals):

       x = data.tensor_dat(data.randn_pos)(
              self.backend, indnames, shape, 
              seed=1, nvals=nvals, defaultval=1+0j
           )
       y = data.tensor_dat(data.randn_pos)(
              self.backend, indnames, shape, 
              seed=2, nvals=nvals, defaultval=1+0j
           )

       xtensor = x.tensor
       ytensor = tn.TensorGen(y.array, x.inds)

       out = tn.notequal(xtensor, ytensor) 
       ans = ar.notequal(x.array, y.array)
       ans = tn.TensorGen(ans, x.inds)

       assert out == ans


   @pytest.mark.parametrize("indnames, shape, nvals", [
      ["ijk", (2,3,4), 5],
   ])
   def test_logical_and(self, indnames, shape, nvals):

       x = data.tensor_dat(data.randn_pos)(
              self.backend, indnames, shape, 
              seed=1, dtype="bool", nvals=nvals, defaultval=False
           )
       y = data.tensor_dat(data.randn_pos)(
              self.backend, indnames, shape, 
              seed=2, dtype="bool", nvals=nvals, defaultval=False
           )

       xtensor = x.tensor
       ytensor = tn.TensorGen(y.array, x.inds)

       out = tn.logical_and(xtensor, ytensor) 
       ans = ar.logical_and(x.array, y.array)
       ans = tn.TensorGen(ans, x.inds)

       assert out == ans


   @pytest.mark.parametrize("indnames, shape, nvals", [
      ["ijk", (2,3,4), 5],
   ])
   def test_logical_or(self, indnames, shape, nvals):

       x = data.tensor_dat(data.randn_pos)(
              self.backend, indnames, shape, 
              seed=1, dtype="bool", nvals=nvals, defaultval=False
           )
       y = data.tensor_dat(data.randn_pos)(
              self.backend, indnames, shape, 
              seed=2, dtype="bool", nvals=nvals, defaultval=False
           )

       xtensor = x.tensor
       ytensor = tn.TensorGen(y.array, x.inds)

       out = tn.logical_or(xtensor, ytensor) 
       ans = ar.logical_or(x.array, y.array)
       ans = tn.TensorGen(ans, x.inds)

       assert out == ans




