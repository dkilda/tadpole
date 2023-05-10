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
import tests.array.data   as ardata


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
###  Tensor space                                                           ###
###                                                                         ###
###############################################################################


# --- TensorSpace ----------------------------------------------------------- #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestTensorSpace:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   # --- Fill space with data --- #

   @pytest.mark.parametrize("inds, shape, dtype", [
      ["ijk", (2,3,4), "complex128"],
   ])
   def test_fillwith(self, inds, shape, dtype):

       w = data.tensorspace_dat(self.backend, inds, shape, dtype)

       x = data.array_dat(data.randn)(
              self.backend, w.shape, dtype=w.dtype
           )
   
       out = w.tensorspace.fillwith(x.array)
       ans = tn.TensorGen(x.array, w.inds)

       assert out == ans


   # --- Reshape space --- #

   @pytest.mark.parametrize("dtype", [
      "complex128"
   ])
   @pytest.mark.parametrize("shape, inds, inds1, inds2", [
      [(2,3,4,5,6,7), "ijklmn", "ijk", "mjlin"],
   ])
   def test_reshape(self, dtype, shape, inds, inds1, inds2):

       v = data.indices_dat(inds, shape)

       inds1 = Indices(*v.inds.map(*inds1))
       inds2 = Indices(*v.inds.map(*inds2))

       x = tn.TensorSpace(
              ar.ArraySpace(backends.get(self.backend), inds1.shape, dtype), 
              inds1
           )

       ans = tn.TensorSpace(
                ar.ArraySpace(backends.get(self.backend), inds2.shape, dtype), 
                inds2
             )

       assert x.reshape(inds2) == ans


   # --- Gradient factories --- #

   @pytest.mark.parametrize("graddat", [
      data.sparsegrad_dat_001,
      data.sparsegrad_dat_002,
      data.sparsegrad_dat_003,
      data.sparsegrad_dat_004,
      data.sparsegrad_dat_005,
   ])
   def test_sparsegrad(self, graddat):

       w = graddat(self.backend)        
       assert w.space.sparsegrad(w.pos, w.vals) == w.grad 


   @pytest.mark.parametrize("graddat", [
      data.nullgrad_dat_001,
   ])
   def test_nullgrad(self, graddat):

       w = graddat(self.backend)        
       assert w.space.nullgrad() == w.grad


   # --- Tensor factories --- #

   @pytest.mark.parametrize("sampledat", [
      ardata.zeros_dat_001,
   ])
   def test_zeros(self, sampledat):

       w = data.tensor_sample_dat(sampledat)(
              self.backend
           )
       x = tn.TensorSpace(
              ar.ArraySpace(w.backend, w.shape, w.dtype), 
              w.inds
           ) 

       assert x.zeros() == w.tensor


   @pytest.mark.parametrize("sampledat", [
      ardata.ones_dat_001,
   ])
   def test_ones(self, sampledat):

       w = data.tensor_sample_dat(sampledat)(
              self.backend
           )
       x = tn.TensorSpace(
              ar.ArraySpace(w.backend, w.shape, w.dtype), 
              w.inds
           ) 

       assert x.ones() == w.tensor


   @pytest.mark.parametrize("basisdat", [
      ardata.basis_real_dat_001,
   ])
   def test_unit(self, basisdat):

       w = data.tensor_basis_dat(basisdat)(
              self.backend
           )
       x = tn.TensorSpace(
              ar.ArraySpace(w.backend, w.shape, w.dtype), 
              w.inds
           ) 

       for pos, tensor in zip(w.pos, w.tensors):
           assert x.unit(pos) == tensor  


   @pytest.mark.parametrize("shape, inds, lind, rind, ldim, rdim", [
      [(2,3),   "ij",  None, None, 2, 3],
      [(2,3,4), "ijk", "i", "k",   2, 4],
   ])
   def test_eye(self, shape, inds, lind, rind, ldim, rdim):

       w = data.tensor_dat(data.randn)(
              self.backend, inds, shape
           )

       x = tn.TensorSpace(
              ar.ArraySpace(w.backend, w.shape, w.dtype), 
              w.inds
           ) 

       out = x.eye(lind, rind)

       if   lind and rind:
            lind, rind = w.inds.map(lind, rind)
       else:
            lind, rind = w.inds
           
       ans = ar.eye(ldim, rdim, dtype=w.dtype, backend=w.backend)
       ans = tn.TensorGen(ans, (lind, rind))

       assert out == ans


   @pytest.mark.parametrize("sampledat", [
      ardata.rand_real_dat_001,
      ardata.rand_complex_dat_001,
   ])
   def test_rand(self, sampledat):

       w = data.tensor_sample_dat(sampledat)(
              self.backend, seed=1
           )
       x = tn.TensorSpace(
              ar.ArraySpace(w.backend, w.shape, w.dtype), 
              w.inds
           ) 

       assert x.rand(seed=1) == w.tensor     


   @pytest.mark.parametrize("sampledat", [
      ardata.randn_real_dat_001,
      ardata.randn_complex_dat_001,
   ])
   def test_randn(self, sampledat):

       w = data.tensor_sample_dat(sampledat)(
              self.backend, seed=1
           )
       x = tn.TensorSpace(
              ar.ArraySpace(w.backend, w.shape, w.dtype), 
              w.inds
           ) 

       assert x.randn(seed=1) == w.tensor        


   @pytest.mark.parametrize("sampledat", [
      ardata.randuniform_real_dat_001,
      ardata.randuniform_complex_dat_001,
   ])
   def test_randuniform(self, sampledat):

       w = data.tensor_sample_dat(sampledat)(
              self.backend, seed=1
           )
       x = tn.TensorSpace(
              ar.ArraySpace(w.backend, w.shape, w.dtype), 
              w.inds
           ) 

       assert x.randuniform(w.opts["boundaries"], seed=1) == w.tensor  


   @pytest.mark.parametrize("basisdat", [
      ardata.basis_real_dat_001,
   ])
   def test_units(self, basisdat):

       w = data.tensor_basis_dat(basisdat)(
              self.backend
           )
       x = tn.TensorSpace(
              ar.ArraySpace(w.backend, w.shape, w.dtype), 
              w.inds
           ) 

       assert tuple(x.units()) == tuple(w.tensors)  


   @pytest.mark.parametrize("basisdat", [
      ardata.basis_real_dat_001,
      ardata.basis_complex_dat_001,
   ])
   def test_basis(self, basisdat):

       w = data.tensor_basis_dat(basisdat)(
              self.backend
           )
       x = tn.TensorSpace(
              ar.ArraySpace(w.backend, w.shape, w.dtype), 
              w.inds
           ) 

       assert tuple(x.basis()) == tuple(w.tensors) 
 

   # --- Space properties --- #

   @pytest.mark.parametrize("inds, shape, dtype", [
      ["ijk", (2,3,4), "complex128"],
   ])
   def test_dtype(self, inds, shape, dtype):

       w = data.tensorspace_dat(self.backend, inds, shape, dtype)

       assert w.tensorspace.dtype == w.dtype


   @pytest.mark.parametrize("inds, shape, dtype", [
      ["ijk", (2,3,4), "complex128"],
   ])
   def test_size(self, inds, shape, dtype):

       w = data.tensorspace_dat(self.backend, inds, shape, dtype)

       assert w.tensorspace.size == np.prod(w.shape)


   @pytest.mark.parametrize("inds, shape, dtype", [
      ["ijk", (2,3,4), "complex128"],
   ])
   def test_ndim(self, inds, shape, dtype):

       w = data.tensorspace_dat(self.backend, inds, shape, dtype)

       assert w.tensorspace.ndim == len(w.shape)


   @pytest.mark.parametrize("inds, shape, dtype", [
      ["ijk", (2,3,4), "complex128"],
   ])
   def test_shape(self, inds, shape, dtype):

       w = data.tensorspace_dat(self.backend, inds, shape, dtype)

       assert w.tensorspace.shape == w.shape




